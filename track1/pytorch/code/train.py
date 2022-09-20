# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:57:34 2022

@author: DELL
"""
import numpy as np
import torch
import warnings
import time
import os
import random
import timm.scheduler as timm_scheduler
from nets.net_training import focal_loss, reg_l1_loss, ohem_bce_dice_loss
from utils.dataloader import RoadDataset, road_dataset_collate
from torch.utils.data import DataLoader
from nets.hrnet import HRnet

# 忽略警告信息
warnings.filterwarnings('ignore')

# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 设置随机种子
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.manual_seed(seed)
    

def train(model, optimizer, scheduler, train_loader, epoches, batch_size, model_path): 
    
    header = r'Epoch/EpochNum | SegLoss | DetLoss | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.5f} | {:9.5f} | {:9.2f}'
    print(header)
 
    # # 记录当前验证集最优IoU,以判定是否保存当前模型
    # best_iou = 0
    # best_iou_epoch = 0
    # train_loss_epochs, val_iou_epochs, lr_epochs = [], [], []
    # 开始训练
    model.train()
    for epoch in range(1, epoches+1):
        # 存储训练集每个batch的loss
        losses = []
        seg_losses = []
        det_losses = []
        
        start_time = time.time()
        for batch_index, batch in enumerate(train_loader):
            
            batch = [ann.to(DEVICE) for ann in batch]
            batch_images, batch_label, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
            
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            seg, hm, wh, offset = model(batch_images)
            # 求解该batch的loss
            seg             = seg.to(torch.float32)
            batch_label     = batch_label.to(torch.float32)
            seg_loss        = ohem_bce_dice_loss(seg, batch_label)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            det_loss        = 0.1*(c_loss + wh_loss + off_loss)
            loss            = seg_loss + det_loss
            losses.append(loss.item())
            seg_losses.append(seg_loss.item())
            det_losses.append(det_loss.item())
            
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            
        scheduler.step(epoch-1)
        # 输出进程
        print(raw_line.format(epoch, epoches, np.array(seg_losses).mean(), 
                              np.array(det_losses).mean(), 
                              (time.time()-start_time)/60))   
        
        torch.save(model.state_dict(), model_path.replace(".pth",f"_{epoch}.pth"))
    

if __name__ == '__main__': 
    
    seed_it(1)
    epoches                 = 50
    batch_size              = 4
    input_shape             = (1024,1024)
    seg_classes             = 1
    det_classes             = 2
    num_workers             = 8
    train_annotation_path   = "train_all.txt"
    
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    
    train_dataset   = RoadDataset(train_lines, input_shape, seg_classes, 
                                  det_classes, train = True)
    
    train_loader    = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, 
                                 num_workers = num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=road_dataset_collate)
    
    model = HRnet(seg_classes = 1, det_classes = 2, 
                  backbone = 'hrnetv2_w48', pretrained = True)
    
    model.to(DEVICE)
    model_path = "../model/hr_w48_fcn_center_1024.pth"
    
    # 采用AdamM优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    # # 余弦退火调整学习率
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10, # T_0就是初始restart的epoch数目
    #     T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
    #     eta_min=1e-5, # 最低学习率
    #     )
    
    scheduler = timm_scheduler.CosineLRScheduler(optimizer, 
                                                 t_initial=100,
                                                 lr_min=1e-5,
                                                 warmup_t=5,
                                                 warmup_lr_init=1e-5)
    
    train(model, optimizer, scheduler, train_loader, epoches, batch_size, model_path)