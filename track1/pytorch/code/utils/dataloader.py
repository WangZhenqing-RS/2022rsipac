# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:31:48 2022

@author: DELL
"""
import math

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input, randomHueSaturationValue, randomHorizontalFlip, randomVerticleFlip, randomRotate90

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class RoadDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, seg_classes, det_classes, train):
        super(RoadDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/4) , int(input_shape[1]/4))
        self.det_classes        = det_classes
        self.seg_classes        = seg_classes
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        image, label, box       = self.get_random_data(self.annotation_lines[index],self.input_shape)
        label[label==255]       = 1
        
        
        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.det_classes), dtype=np.float32)
        batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)

        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                #---------------------------------------------------#
                #   计算宽高真实值
                #---------------------------------------------------#
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                #---------------------------------------------------#
                #   计算中心偏移量
                #---------------------------------------------------#
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                #---------------------------------------------------#
                #   将对应的mask设置为1
                #---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0]] = 1
                
        
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomHorizontalFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomVerticleFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomRotate90(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        
        image = np.transpose(preprocess_input(image), (2, 0, 1))
        label = np.expand_dims(label,0)
        label = np.array(label, np.int64)
        return image.copy(), label.copy(), batch_hm.copy(), batch_wh.copy(), batch_reg.copy(), batch_reg_mask.copy()
    

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        
        image_path  = line[0]
        image       = cv2.imread(image_path)
        
        image       = randomHueSaturationValue(image,
                                               hue_shift_limit=(-30, 30),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
        
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #------------------------------#
        #   读取分割标签
        #------------------------------#
        
        label_path  = line[1]
        label       = cv2.imread(label_path,0)
        
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
        
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih, iw  = label.shape
        h, w    = input_shape
        
        if iw!=w:
            image = cv2.resize(image,input_shape)
            label = cv2.resize(label,input_shape,interpolation=cv2.INTER_NEAREST)
            if len(box)!=0:
                # min_x,min_y,max_x,max_y 缩放
                box = 1.0 * box / ih * h
                # 类别 0*$ -> 0; 1*$ -> $
                box[:,-1][box[:,-1]==1.0/ih*h] = 1
                box = np.uint16(box)
            
        
        # image_data  = np.array(image, np.float32)
        # label_data  = np.array(label, np.int64)
        # return image_data, label_data, box
        return image, label, box
                

# DataLoader中collate_fn使用
def road_dataset_collate(batch):
    imgs, labels, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], [], []

    for img, label, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        labels.append(label)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs            = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    labels          = torch.from_numpy(np.array(labels)).long()
    batch_hms       = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_whs       = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs      = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    return imgs, labels, batch_hms, batch_whs, batch_regs, batch_reg_masks