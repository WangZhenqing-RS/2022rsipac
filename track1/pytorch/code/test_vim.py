# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:26:04 2022

@author: DELL
"""

import os
import json
import numpy as np
import cv2
import glob
import tqdm

def cv_imread(file_name):
    # 用于解决 cv2 无法直接读取路径含有中文的图片
    
    cv_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
    return cv_img


# -----------------------------------------------------------------------------
# --------------------- COCO 标注的大字典里找那张图片的信息 ---------------------
# -----------------------------------------------------------------------------
def get_COCO_img_info(img_name, all_coco_ann):
    
    # 从 COCO 标注的那个大字典里 找 img_name 的名字
    # 找到了就返回, 没找到就 return False
    
    for img_info in all_coco_ann["images"]:
        if img_info['file_name'] == img_name:
            return img_info
        else:
            continue
    return False 


# -----------------------------------------------------------------------------
# --------------------- COCO 标注的大字典里找那张图片的标注 ---------------------
# -----------------------------------------------------------------------------
def get_COCO_img_anno(img_id, all_coco_ann):
    
    # 根据图片的 id 找标注的信息
    # 找到了就返回那个列表, 没找到就 return []
    
    ann_list = []
    for ann_info in all_coco_ann["annotations"]:
        if ann_info['image_id'] == img_id:
            ann_list.append(ann_info)
        else:
            continue
    return ann_list 


# -----------------------------------------------------------------------------
# ------------------------- 获取你想要的的类别的类别id  ------------------------
# -----------------------------------------------------------------------------
def get_categories_needed(category, all_coco_ann):
    
    # category 可以使一个类(字符串) 也可以是好几个类(字符串的列表)
    if isinstance(category, str):
        category = [category]
    
    cls_id2name = {}
    cls_name2id = {}
    for cls_info in all_coco_ann["categories"]:
        if cls_info['name'] in category:
            cls_id2name[cls_info['id']] = cls_info['name']
            cls_name2id[cls_info['name']] = cls_info['id']

    return cls_id2name, cls_name2id



# -----------------------------------------------------------------------------
# ---------------------- 根据已选择的类别挑选已获得的标注  ----------------------
# -----------------------------------------------------------------------------
def get_ann_needed(ann_list, cls_id2name):
    
    # 根据标注列表 ann_list 和 需要的类别字典 cls_id2name
    
    ann_you_want = []
    for ann in ann_list:
        if ann['category_id'] in cls_id2name:
            ann_you_want.append( (cls_id2name[ann['category_id']], ann['bbox']) )
    return ann_you_want



# -----------------------------------------------------------------------------
# -------------------------------- 读图绘制bbox  -------------------------------
# -----------------------------------------------------------------------------
def drawBbox(img_array, ann_needed, color=(0, 255, 0)):
    
    # 在图片上绘制 bbox
    
    # 我没想到下面这句话这么重要！！ 后面由于是传的引用会直接在原图上操作
    img_array = img_array.copy() 
    
    for name, (x_lt, y_lt, w, h) in ann_needed:
        img_array = cv2.rectangle(img_array, 
                                  (int(x_lt),   int(y_lt)),
                                  (int(x_lt+w), int(y_lt+h)),
                                  # (int(y_lt),   int(x_lt)),
                                  # (int(y_lt+h), int(x_lt+w)),
                                  color, # 这里可以根据类别自己换颜色
                                  3)
    
    return img_array

with open(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\instances_test.json", encoding="utf-8") as f:
    test_image_infos = json.loads(f.read())
with open(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\results\test.bbox.json", encoding="utf-8") as f:
    test_annos = json.loads(f.read())
    
    
test_image_infos["annotations"] = test_annos



image_paths = glob.glob(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\images/*.tif")
save_folder = r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\images_ann__"
if not os.path.exists(save_folder): os.makedirs(save_folder)


for image_path in tqdm.tqdm(image_paths):
    img_name = os.path.basename(image_path)
    
    img_info = get_COCO_img_info(img_name, test_image_infos)
    img_id = img_info['id']
    
    category = ['cross']
    cls_id2name, cls_name2id = get_categories_needed(category, test_image_infos)
    ann_list = get_COCO_img_anno(img_id, test_image_infos)
    ann_needed = get_ann_needed(ann_list, cls_id2name)
    img_array = cv_imread(image_path)
    img_array = drawBbox(img_array, ann_needed)
    
    
    category = ['uncross']
    cls_id2name, cls_name2id = get_categories_needed(category, test_image_infos)
    ann_list = get_COCO_img_anno(img_id, test_image_infos)
    ann_needed = get_ann_needed(ann_list, cls_id2name)
    img_array = drawBbox(img_array, ann_needed, color=(0,0,255))
    
    cv2.imwrite(os.path.join(save_folder, img_name),img_array)
    # break
