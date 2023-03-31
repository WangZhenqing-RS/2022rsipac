# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:14:11 2022

@author: DELL
"""
import cv2
import glob
import tqdm
import os


# '''
# 生成train.txt val.txt
# '''
# def gen_train_txt(annotations_crop_dir, train_all_txt_path):
    
#     txt_paths = sorted(glob.glob(annotations_crop_dir+"/*.txt"))
    
#     if os.path.exists(train_all_txt_path): os.remove(train_all_txt_path)

#     # 打开要写的train_all_txt_path
#     train_all_txt_f = open(train_all_txt_path, 'a')

#     for i, txt_path in tqdm.tqdm(enumerate(txt_paths)):
#         txt_path = os.path.abspath(txt_path)
#         image_path = txt_path.replace("annotations_crop", "images_crop").replace("txt", "tif")
#         label_path = txt_path.replace("annotations_crop", "masks_crop").replace("txt", "png")
#         train_all_txt_f.write(image_path+" "+label_path)

#         with open(txt_path, "r") as f:  # 打开文件
#             data = f.read()  # 读取文件
#         annotations = data.split("\n")
#         for annotation in annotations:
#             annotation = annotation.split(" ")
#             if len(annotation)<5:
#                 continue
#             min_x,min_y,max_x,max_y,category = annotation
#             train_all_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
#         train_all_txt_f.write('\n')

#     train_all_txt_f.close()
    
    
'''
生成train.txt val.txt
'''
def gen_train_txt(annotations_crop_dir, annotations_crop_dir2, train_all_txt_path):
    
    txt_paths = sorted(glob.glob(annotations_crop_dir+"/*.txt"))
    
    if os.path.exists(train_all_txt_path): os.remove(train_all_txt_path)

    # 打开要写的train_all_txt_path
    train_all_txt_f = open(train_all_txt_path, 'a')

    for i, txt_path in tqdm.tqdm(enumerate(txt_paths)):
        txt_path = os.path.abspath(txt_path)
        image_path = txt_path.replace("annotations_crop", "images_crop").replace("txt", "tif")
        label_path = txt_path.replace("annotations_crop", "masks_crop").replace("txt", "png")
        train_all_txt_f.write(image_path+" "+label_path)

        with open(txt_path, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
        annotations = data.split("\n")
        for annotation in annotations:
            annotation = annotation.split(" ")
            if len(annotation)<5:
                continue
            min_x,min_y,max_x,max_y,category = annotation
            train_all_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
        train_all_txt_f.write('\n')
    
    txt_paths = sorted(glob.glob(annotations_crop_dir2+"/*.txt"))
    for i, txt_path in tqdm.tqdm(enumerate(txt_paths)):
        txt_path = os.path.abspath(txt_path)
        image_path = txt_path.replace("annotations_crop", "images_crop").replace("txt", "tif")
        label_path = txt_path.replace("annotations_crop", "masks_crop").replace("txt", "png")
        train_all_txt_f.write(image_path+" "+label_path)

        with open(txt_path, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
        annotations = data.split("\n")
        for annotation in annotations:
            annotation = annotation.split(" ")
            if len(annotation)<5:
                continue
            min_x,min_y,max_x,max_y,category = annotation
            train_all_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
        train_all_txt_f.write('\n')
        
    train_all_txt_f.close()
    
if __name__ == "__main__":
    
    
    annotations_crop_dir = '../../data/fusai_release/train/annotations_crop'
    train_all_txt_path = 'code_mindspore/train_all.txt'
    gen_train_txt(annotations_crop_dir, train_all_txt_path)