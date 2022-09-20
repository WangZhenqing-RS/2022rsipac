# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:04:40 2022

@author: DELL
"""
import cv2
import numpy as np
import glob
import tqdm

image_paths = glob.glob(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\train\images/*.tif")
image_num = len(image_paths)
images = []
for image_path in tqdm.tqdm(image_paths):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(int(image.shape[0]/4),int(image.shape[1]/4)))
    # BGR2RGB
    image = image.astype(np.float32)[:, :, ::-1]  
    # HWC2CHW
    image = image.transpose((2, 0, 1))              
    image = image / 255.0
    # 图像尺寸不一，需要拉直后再拼接
    image = image.reshape(3,image.shape[1]*image.shape[2])
    images.append(image)

images = np.concatenate(images,axis=1)

images_mean = [np.mean(images[0]), np.mean(images[1]), np.mean(images[2])]
image_std = [np.std(images[0]), np.std(images[1]), np.std(images[2])]
print("images_mean")
# [0.38983384, 0.41159928, 0.3616227]
print(images_mean)
print("image_std")
# [0.20442073, 0.17412226, 0.16636926]
print(image_std)