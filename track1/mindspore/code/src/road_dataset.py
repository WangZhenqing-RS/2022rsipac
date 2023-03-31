# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Dataset Cityscapes generator."""
import cv2
import math
import numpy as np

import mindspore.ops as P
from mindspore import Tensor
from mindspore.common import dtype

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

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomHorizontalFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        batch_hm = cv2.flip(batch_hm, 1)
        batch_wh = cv2.flip(batch_wh, 1)
        batch_reg = cv2.flip(batch_reg, 1)
        batch_reg_mask = cv2.flip(batch_reg_mask, 1)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def randomVerticleFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
        batch_hm = cv2.flip(batch_hm, 0)
        batch_wh = cv2.flip(batch_wh, 0)
        batch_reg = cv2.flip(batch_reg, 0)
        batch_reg_mask = cv2.flip(batch_reg_mask, 0)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def randomRotate90(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        label=np.rot90(label)
        batch_hm=np.rot90(batch_hm)
        batch_wh=np.rot90(batch_wh)
        batch_reg=np.rot90(batch_reg)
        batch_reg_mask=np.rot90(batch_reg_mask)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def preprocess_input(image):
    mean    = [0.38983384, 0.41159928, 0.3616227]
    std     = [0.20442073, 0.17412226, 0.1663693]
    return (image / 255. - mean) / std

class RoadDataset():
    
    def __init__(self,
                 annotation_lines, 
                 input_shape, 
                 seg_classes, 
                 det_classes):

        super().__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/4) , int(input_shape[1]/4))
        self.det_classes        = det_classes
        self.seg_classes        = seg_classes

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
             
        # 数据增强有问题，尚未解决
        # print(image.shape, label.shape, batch_hm.shape, batch_wh.shape, batch_reg.shape, batch_reg_mask.shape)
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomHorizontalFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomVerticleFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask = randomRotate90(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask)
        
        # # 将一个内存不连续存储的数组转换为内存连续存储的数组
        # image = np.ascontiguousarray(image)
        # label = np.ascontiguousarray(label)
        # batch_hm = np.ascontiguousarray(batch_hm)
        # batch_wh = np.ascontiguousarray(batch_wh)
        # batch_reg = np.ascontiguousarray(batch_reg)
        # batch_reg_mask = np.ascontiguousarray(batch_reg_mask)
        
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
        
        # # HSV增强
        # image       = randomHueSaturationValue(image,
        #                                        hue_shift_limit=(-30, 30),
        #                                        sat_shift_limit=(-5, 5),
        #                                        val_shift_limit=(-15, 15))
        
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
        
        # 图像标签缩放
        if iw!=w:
            image = cv2.resize(image,input_shape)
            label = cv2.resize(label,input_shape,interpolation=cv2.INTER_NEAREST)
            if len(box)!=0:
                # min_x,min_y,max_x,max_y 缩放
                box = 1.0 * box / ih * h
                # 类别 0*$ -> 0; 1*$ -> $
                box[:,-1][box[:,-1]==1.0/ih*h] = 1
                box = np.uint16(box)
            
        return image, label, box
    
if __name__ == "__main__":
    train_annotation_path   = "code_mindspore/train_all.txt"
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    input_shape = (1024,1024)
    seg_classes = 1
    det_classes = 2
    dategen = RoadDataset(train_lines,
                          input_shape, 
                          seg_classes, 
                          det_classes)

    for i, data in enumerate(dategen):
        image, label, _, _, _, _ = data
        print(image.max())
        # break