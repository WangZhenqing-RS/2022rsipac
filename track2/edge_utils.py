# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:24:41 2021

@author: DELL
"""

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        # 提取轮廓
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = (edgemap > 0).astype(np.uint8)*255
    return edgemap


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

# if __name__ == '__main__':
#     label = cv2.imread(r"E:\WangZhenQing\2021GaoFen\code\41_1_label.png",0)
#     label[label==255] = 1
#     print(np.unique(label))
#     img = cv2.imread(r"E:\WangZhenQing\2021GaoFen\code\41_1.png")
#     oneHot_label = mask_to_onehot(label, 2)
#     edge = onehot_to_binary_edges(oneHot_label, 2, 2) # #edge=255,background=0
#     edge[:2, :] = 0
#     edge[-2:, :] = 0
#     edge[:, :2] = 0
#     edge[:, -2:] = 0
#     # print(edge)
#     print(np.unique(edge))
#     print(edge.shape)
#     cv2.imwrite('test.png',edge)