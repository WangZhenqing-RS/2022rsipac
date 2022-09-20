# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:54:59 2022

@author: DELL
"""
# import math
# from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn
# from segmentation_models_pytorch.losses import DiceLoss

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def dice_loss(pred, target):
    
    epsilon = 1e-5
    num = pred.size(0)
    
    pred = pred.view(num, -1)
    target = target.view(num, -1)
    
    # 利用预测值与标签相乘当作交集
    intersection = (pred * target).sum(-1).sum()  
    union = (pred + target).sum(-1).sum()
    
    score = 1 - 2 * (intersection + epsilon) / (union + epsilon)
    
    return score
    
def ohem_bce_dice_loss(pred, target):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    # dice_loss = DiceLoss(mode='binary',
    #                      from_logits=False)
    # 交叉熵
    bce_loss = nn.BCELoss(reduction='none')
    
    loss_bce = bce_loss(pred, target)
    loss_dice = dice_loss(pred, target)
    
    # OHEM
    loss_bce_,ind = loss_bce.contiguous().view(-1).sort()
    min_value = loss_bce_[int(0.5*loss_bce.numel())]
    loss_bce = loss_bce[loss_bce>=min_value]
    loss_bce = loss_bce.mean()
    loss = loss_bce + loss_dice
    
    return loss
