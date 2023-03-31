# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 20:51:41 2022

@author: DELL
"""
import cv2
import os
import glob
import shutil
import tqdm
import json
import numpy as np
from skimage import io


def get_imagepaths_from_json(json_path, image_folder):
    
    image_paths = []
    with open(json_path, encoding="utf-8") as f:
        annotations_test = json.load(f)
    
    images_info = annotations_test["images"]
    images_num = len(images_info)
    for i in range(images_num):
        image_path = os.path.join(image_folder, images_info[i]["file_name"])
        image_paths.append(image_path)
    return image_paths

#  图像宽高不足裁剪宽高度,填充至裁剪宽高度
def fill_right_bottom(img, size_w, size_h, value=(0,0,0)):
    size = img.shape
    img_fill_right_bottom = cv2.copyMakeBorder(img, 0, max(0,size_h - size[0]), 0, max(0,size_w - size[1]), 
                                               cv2.BORDER_CONSTANT, value = value)
    return img_fill_right_bottom


def data_clip(json_path, img_floder, label_folder, anno_folder, out_img_floder, 
              out_label_floder, out_anno_floder,
              size_w = 1024, size_h = 1024, step = 1000):
    
    if os.path.exists(out_img_floder):
        shutil.rmtree(out_img_floder)
    os.makedirs(out_img_floder)
    if os.path.exists(out_label_floder):
        shutil.rmtree(out_label_floder)
    os.makedirs(out_label_floder)
    if os.path.exists(out_anno_floder):
        shutil.rmtree(out_anno_floder)
    os.makedirs(out_anno_floder)
    
    with open(json_path, encoding="utf-8") as f:
        annotations_test = json.load(f)
    images_info = annotations_test["images"]
    images_num = len(images_info)
    
    count = 0
    for i in tqdm.tqdm(range(images_num)):
        
        img_path = os.path.join(img_floder, images_info[i]["file_name"])
        number = 0
        
        img_name = os.path.basename(img_path)[:-4]
        label_path = os.path.join(label_folder,img_name+".png")
        anno_path = os.path.join(anno_folder,img_name+".txt")
        
        # img = cv2.imread(img_path)
        # label = cv2.imread(label_path,0)
        # # 便于可视化
        # label[label!=0] = 255
        # anno_file = open(anno_path, 'r') 
        # anno_lines = anno_file.readlines()
        # print(img_path)
        try:
            img = io.imread(img_path)
            img = img[:,:,0:3]
            img = np.uint8(img)
            label = cv2.imread(label_path,0)
            # 便于可视化
            label[label!=0] = 255
            anno_file = open(anno_path, 'r') 
            anno_lines = anno_file.readlines()
        except:
            h = images_info[i]["height"]
            w = images_info[i]["width"]
            img = np.zeros((h,w,3),np.uint8)
            label = np.zeros((h,w),np.uint8)
            anno_lines = []
            # print(f"{os.path.basename(img_path)} is error, replace with zeros.")

        size = img.shape
        
        # 
        if size[0] < size_h or size[1] < size_w:
            # print(f'图片{img_name}需要补齐')
            img = fill_right_bottom(img,  size_w, size_h)
            label = fill_right_bottom(label,  size_w, size_h, value=(0))
        
        size = img.shape
        
        count = count + 1
        for h in range(0, size[0] - 1, step):
            start_h = h
            for w in range(0, size[1] - 1, step):
                
                start_w = w
                end_h = start_h + size_h
                if end_h > size[0]:
                   start_h = size[0] - size_h 
                   end_h = start_h + size_h
                end_w = start_w + size_w
                if end_w > size[1]:
                   start_w = size[1] - size_w
                end_w = start_w + size_w
                
                img_cropped = img[start_h : end_h, start_w : end_w]
                label_cropped = label[start_h : end_h, start_w : end_w]
                
                # 含有正例才保存       22405
                # 正例占比>0.01才保存    
                # if np.max(label_cropped)!=0:
                if (np.sum(label_cropped==255)/size_h/size_h) > 0.01:
                    #  用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
                    img_name_cropped = img_name + '_'+ str(start_h) +'_' + str(start_w)
                    
                    cv2.imwrite(os.path.join(out_img_floder,img_name_cropped+".tif"), img_cropped)
                    cv2.imwrite(os.path.join(out_label_floder,img_name_cropped+".png"), label_cropped)
                    
                    
                    anno_cropped_file = open(os.path.join(out_anno_floder,img_name_cropped+".txt"), 'a')
                    
                    
                    # 逐行读取
                    for anno_line in anno_lines:
                        anno_line_split = anno_line.split(' ')
                        min_x,min_y,max_x,max_y,category = anno_line_split
                        
                        # 裁剪后的box对应原图的坐标
                        cropped_min_x = max(start_w,int(min_x))
                        cropped_max_x = min(end_w,int(max_x))
                        cropped_min_y = max(start_h,int(min_y))
                        cropped_max_y = min(end_h,int(max_y))
                        
                        area_cropped = (cropped_max_y-cropped_min_y)*(cropped_max_x-cropped_min_x)
                        area_box = (int(max_y)-int(min_y))*(int(max_x)-int(min_x))
                        
                        if cropped_max_y-cropped_min_y>0 and area_cropped/area_box>=0.5:
                            # print(area_cropped, area_box, area_cropped/area_box)
                            min_x_cropped = cropped_min_x-start_w
                            min_y_cropped = cropped_min_y-start_h
                            max_x_cropped = cropped_max_x-start_w
                            max_y_cropped = cropped_max_y-start_h
                            anno_cropped_file.write(f"{min_x_cropped} {min_y_cropped} {max_x_cropped} {max_y_cropped} {category}")
                    
                    number = number + 1
                
        anno_file.close()
        anno_cropped_file.close()     

        # print('{}.png切割成{}张.'.format(img_name,number))
    # print('共完成{}张图片'.format(count))
    
if __name__ == '__main__':

    # 图像、标签和注解
    json_path = "../../data/fusai_release/train/instances_train.json"
    img_floder = '../../data/fusai_release/train/images'
    label_folder = '../../data/fusai_release/train/masks'
    anno_folder = '../../data/fusai_release/train/annotations'
    
    # 裁剪的图像、标签和注解
    out_img_floder = '../../data/fusai_release/train/images_crop'
    out_label_floder = '../../data/fusai_release/train/masks_crop'
    out_anno_floder = '../../data/fusai_release/train/annotations_crop'
    
    
    data_clip(json_path, img_floder, label_folder, anno_folder, out_img_floder, 
              out_label_floder, out_anno_floder)