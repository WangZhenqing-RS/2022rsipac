# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:11:32 2022

@author: DELL
"""

import numpy as np
from utils.utils import preprocess_input
from utils.utils_bbox import decode_bbox, postprocess
import torch
from nets.hrnet import HRnet
import warnings
# from PIL import Image
import cv2
import json
import os
import tqdm


# 忽略警告信息
warnings.filterwarnings('ignore')

# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def infer(model, image, confidence=0.3):
    
    image_copy = image.copy()
    image_copy = np.array(image_copy)
    
    #--------------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #--------------------------------------------------------------------------#
    confidence      = confidence
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    nms_iou         = 0.3
    #--------------------------------------------------------------------------#
    #   是否进行非极大抑制，可以根据检测效果自行选择
    #--------------------------------------------------------------------------#
    nms             = True
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    cuda            = True
    
    
    image_shape = image.shape[0:2]
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)

    model.to(DEVICE)
    with torch.no_grad():
        images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        
        images = images.to(DEVICE)
        
        outputs = model(images)
        
        
        seg_outputs = outputs[0].cpu().data.numpy().squeeze()
        
        seg_outputs[seg_outputs>=0.5] = 255
        seg_outputs[seg_outputs<0.5]  = 0
        seg_outputs = seg_outputs.astype(np.uint8)
        
        
        #-----------------------------------------------------------#
        #   利用预测结果进行解码
        #-----------------------------------------------------------#
        outputs = decode_bbox(outputs[1], 
                              outputs[2], 
                              outputs[3], 
                              confidence, cuda)

        #-------------------------------------------------------#
        #   对于centernet网络来讲，确立中心非常重要。
        #   对于大目标而言，会存在许多的局部信息。
        #   此时对于同一个大目标，中心点比较难以确定。
        #   使用最大池化的非极大抑制方法无法去除局部框
        #   所以还是需要进行非极大抑制的代码
        #-------------------------------------------------------#
        results = postprocess(outputs, nms, image_shape, nms_iou)
        
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            
            return seg_outputs, []

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

    
    det_outputs = []
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        label           = int(c)
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image_shape[0], np.floor(bottom).astype('int32'))
        right   = min(image_shape[1], np.floor(right).astype('int32'))
        
        # det_outputs.append([top,left,bottom-top,right-left,label+1, round(score,4)])
        det_outputs.append([left,top,right-left,bottom-top,label+1, round(score,4)])
        # cv2.rectangle(image_copy, (top,left),(bottom,right),(255,255,255),3)
        # print(top, left, bottom, right, label, score)
        
    return seg_outputs,det_outputs




test_json_path = r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\instances_test.json"
test_image_folder = r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\images"
model_path = r"E:\WangZhenQing\2022HongTu\project_road\model\hr_w48_fcn_center_1024_48.pth"

seg_save_folder = r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test\results\masks"

resize_scale = 1

model = HRnet(seg_classes = 1, det_classes = 2, 
              backbone = 'hrnetv2_w48', pretrained = False)
model.load_state_dict(torch.load(model_path))

with open(test_json_path, encoding="utf-8") as f:
    annotations_test = json.load(f)

images_info = annotations_test["images"]
images_num = len(images_info)

annotation_id = 0

for i in tqdm.tqdm(range(images_num)):
    image_name = images_info[i]["file_name"]
    image_id = images_info[i]["id"]
    image_path = os.path.join(test_image_folder,image_name)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_shape = image.shape[0:2]
    resize_shape = [int(shp*resize_scale) for shp in image_shape]
    
    image = cv2.resize(image, (resize_shape[1],resize_shape[0]))
    
    seg_outputs, det_outputs = infer(model, image, confidence=0.01)
    
    seg_outputs = cv2.resize(seg_outputs, (image_shape[1],image_shape[0]))
    seg_save_path = os.path.join(seg_save_folder,image_name.replace("tif","png"))
    cv2.imwrite(seg_save_path,seg_outputs)
    
    for det_output in det_outputs:
        annotations_test["annotations"].append({
            "area": int(det_output[2]/resize_scale*det_output[3]/resize_scale),
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [int(_/resize_scale) for _ in det_output[:4]],
            "category_id": int(det_output[4]),
            "score": float(det_output[5]),
            "id": annotation_id,
            "ignore": 0,
            "segmentation": []
            })
        annotation_id += 1
    # break
        
with open(r"E:\WangZhenQing\2022HongTu\project_road\data\chusai_release\test/results/test.bbox.json", "w") as f:
    json.dump(annotations_test["annotations"], f)


    