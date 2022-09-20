# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:14:11 2022

@author: DELL
"""
import cv2
import glob
import tqdm
import os


'''
生成train.txt val.txt
'''

txt_paths = sorted(glob.glob(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\train\annotations_crop\*.txt"))

fold = 0
train_txt_path = f'../train_{fold}.txt'
val_txt_path = f'../val_{fold}.txt'
train_all_txt_path = '../train_all.txt'

if os.path.exists(train_txt_path): os.remove(train_txt_path)
if os.path.exists(val_txt_path): os.remove(val_txt_path)
if os.path.exists(train_all_txt_path): os.remove(train_all_txt_path)

# 打开要写的train_txt_path
train_txt_f = open(train_txt_path, 'a')
val_txt_f = open(val_txt_path, 'a')
train_all_txt_f = open(train_all_txt_path, 'a')

for i, txt_path in enumerate(txt_paths):
    image_path = txt_path.replace("annotations_crop", "images_crop").replace("txt", "tif")
    label_path = txt_path.replace("annotations_crop", "masks_crop").replace("txt", "png")
    
    train_all_txt_f.write(image_path+" "+label_path)
    if i%5!=fold: 
        # 在train_txt_path中写入图像路径
        train_txt_f.write(image_path+" "+label_path)
    else:
        val_txt_f.write(image_path+" "+label_path)
        
    with open(txt_path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
    annotations = data.split("\n")
    for annotation in annotations:
        
        annotation = annotation.split(" ")
        
        if len(annotation)<5:
            continue
        min_x,min_y,max_x,max_y,category = annotation
        
        train_all_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
        if i%5!=fold: 
            train_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
        else: 
            val_txt_f.write(f' {min_x},{min_y},{max_x},{max_y},{category}')
    train_all_txt_f.write('\n')
    if i%5!=fold:
        train_txt_f.write('\n')
    else:
        val_txt_f.write('\n')