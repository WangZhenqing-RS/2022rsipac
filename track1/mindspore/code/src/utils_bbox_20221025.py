# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:01:00 2022

@author: DELL
"""
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


def pool_nms(heat, kernel = 3):
    
    heat = ops.Cast()(heat, mindspore.float16)
    hmax = nn.MaxPool2d((kernel, kernel), stride=1, pad_mode="same")(heat)
    keep = ops.Cast()((hmax == heat), mindspore.float16)
    return heat * keep

def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print("pool完毕")
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        
        view_lenth = output_h * output_w
        
        heat_map    = (ops.Transpose()(pred_hms[batch],(1, 2, 0))).view((view_lenth, c))
        pred_wh     = (ops.Transpose()(pred_whs[batch],(1, 2, 0))).view((view_lenth, 2))
        pred_offset = (ops.Transpose()(pred_offsets[batch],(1, 2, 0))).view((view_lenth, 2))

        yv, xv      = ops.Meshgrid()((mindspore.numpy.arange(0, output_h), mindspore.numpy.arange(0, output_w)))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv      = ops.Cast()(xv.flatten(), mindspore.float16), ops.Cast()(yv.flatten(), mindspore.float16)

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        # class_conf, class_pred  = torch.max(heat_map, dim = -1)
        class_pred, class_conf  = ops.ArgMaxWithValue(axis=-1)(heat_map)
        
        mask                    = class_conf > confidence
        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#

        pred_wh_mask        = ops.MaskedSelect()(pred_wh, ops.Concat(1)((ops.Reshape()(mask,(-1,1)),ops.Reshape()(mask,(-1,1)))))
        pred_offset_mask    = ops.MaskedSelect()(pred_offset, ops.Concat(1)((ops.Reshape()(mask,(-1,1)),ops.Reshape()(mask,(-1,1)))))
        
        
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------# 
        xv_mask = ops.ExpandDims()(ops.MaskedSelect()(xv, mask) + pred_offset_mask[..., 0], -1)
        yv_mask = ops.ExpandDims()(ops.MaskedSelect()(yv, mask) + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        bboxes = ops.Concat(axis=1)([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h])
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        
        detect = ops.Concat(axis=-1)([ops.Cast()(bboxes, mindspore.float16), 
                                      ops.Cast()(ops.ExpandDims()(ops.MaskedSelect()(class_conf,mask),-1), mindspore.float16), 
                                      ops.Cast()(ops.ExpandDims()(ops.MaskedSelect()(class_pred, mask),-1), mindspore.float16)])
        detects.append(detect)

    return detects

# def bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#         计算IOU
#     """
#     if not x1y1x2y2:
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#     else:
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)

#     inter_area = ops.clip_by_value(inter_rect_x2 - inter_rect_x1, clip_value_min=0, clip_value_max=1e6) * \
#                  ops.clip_by_value(inter_rect_y2 - inter_rect_y1, clip_value_min=0, clip_value_max=1e6)
                 
#     b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#     b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
#     iou = inter_area / ops.clip_by_value(b1_area + b2_area - inter_area, clip_value_min=1e-6, clip_value_max=1e6)

#     return iou

def centernet_correct_boxes(box_xy, box_wh, image_shape):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    image_shape = np.array(image_shape)
    

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction, need_nms, image_shape, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    
    #----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        
        # print("detections", detections)
        if len(detections) == 0:
            continue
        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels   = np.unique(detections[:, -1])
        
        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            
            print("detections.shape", detections.shape)
            detections_class = detections[detections[:, -1] == c]
            print("detections_class.shape", detections_class.shape)
            if need_nms:
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                # 在Ascend平台上，边界框的分数将被忽略，仅根据框之间的IOU来选择框。
                # 这意味着如果要删除分数较低的框，则需要提前按分数对输入框进行降序排序
                sort_value, sort_index = ops.Sort(descending=True)(detections_class[:,4])
                detections_class = detections_class[sort_index]

                output_boxes, indices, mask = ops.NMSWithMask(nms_thres)(
                    detections_class[:,:5]
                )

                indices_np = indices.asnumpy()
                keep = indices_np[mask.asnumpy()].tolist()
                max_detections = detections_class[keep]


                # #------------------------------------------#
                # #   按照存在物体的置信度排序
                # #------------------------------------------#
                # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # #------------------------------------------#
                # #   进行非极大抑制
                # #------------------------------------------#
                # max_detections = []
                # while detections_class.size(0):
                #     #---------------------------------------------------#
                #     #   取出这一类置信度最高的，一步一步往下判断。
                #     #   判断重合程度是否大于nms_thres，如果是则去除掉
                #     #---------------------------------------------------#
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # #------------------------------------------#
                # #   堆叠
                # #------------------------------------------#
                # max_detections = torch.cat(max_detections).data
            else:
                max_detections  = detections_class
            
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i]
            # print(output[i])
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, image_shape)
    return output