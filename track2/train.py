# -*- coding: utf-8 -*-
"""
@author: wangzhenqing

"""
import torch
import warnings
import glob
import os
import time
import random

import numpy as np
import timm.scheduler as timm_scheduler
import segmentation_models_pytorch as smp
import torch.nn.functional as F

from model import Res_Unet
from edgeBCE_Dice_loss import edgeBCE_Dice_loss
from dataProcess import get_dataloader

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
    
    
def train(epoches, batch_size, train_loader, model, optimizer, scheduler, 
          loss_fn, model_path):
    
    header = r'epoch/epochNum | trainLoss | learnRate |   time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.5f} | {:9.5f} | {:9.2f}'
    print(header)
    
    model.train()
        
    start_epoch = 1
    
    # 开始训练
    for epoch in range(start_epoch, epoches+1):
        start_time = time.time()
        
        # 存储训练集每个batch的loss
        losses = []
        
        for batch_index, (image, target, edge) in enumerate(train_loader):
            
            image, target, edge= image.to(DEVICE), target.to(DEVICE), edge.to(DEVICE)
            
            # scale_list = [0.75, 0.875, 1, 1.125, 1.25]
            scale_list = [0.75, 0.875, 1, 1.125]
            scale = random.choice(scale_list)
            
            image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
            target = F.interpolate(target.float(), scale_factor=scale, mode='nearest').long()
            edge = F.interpolate(edge.float(), scale_factor=scale, mode='nearest').int()

            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            output = model(image)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            # 求解该batch的loss
            loss = loss_fn(output, target, edge)
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            
            losses.append(loss.item())
        
        scheduler.step(epoch-1)

        if epoch>40:
            torch.save(model.state_dict(), model_path.replace(".pth",f"_{epoch}.pth"))
        
        # 输出进程
        print(raw_line.format(epoch, epoches, np.array(losses).mean(),
                              optimizer.param_groups[0]['lr'],
                              (time.time()-start_time)/60**1))
    
# 不加主函数这句话的话,Dataloader多线程加载数据会报错
if __name__ == '__main__':
    
    seed = 45
    seed_it(seed)
    
    epoches = 45
    batch_size = 4
    
    channels = 6
    
    # 定义模型
    model = Res_Unet(in_channels = channels, 
                      n_classes = 1, 
                      # backbone = 'resnet18_ibn_a',
                      # backbone = "resnet34_ibn_a",
                      backbone = "resnet50_ibn_a",
                      # model_path = 'resnet18_ibn_a-2f571257.pth', 
                      # model_path = "resnet34_ibn_a-94bc1577.pth",
                      model_path = "resnet50_ibn_a-d9d0bb7b.pth",
                      # backbone = 'resnet18_ibn_b',
                      # model_path = 'resnet18_ibn_b-bc2f3c11.pth', 
                      dropout_rate = 0.5,
                      scse = True, 
                      use_mish=False, 
                      db_block=False, 
                      hypercolumn=False)
    model.to(DEVICE)
    # model.load_state_dict(torch.load("../model/Unet_resnet50a_dropout5_45.pth"))

    # 采用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    
    scheduler = timm_scheduler.CosineLRScheduler(optimizer, 
                                                 t_initial=100,
                                                 lr_min=1e-5,
                                                 warmup_t=5,
                                                 warmup_lr_init=1e-5)
    
    loss_fn = edgeBCE_Dice_loss
    
    image_A_paths = sorted(glob.glob("/data/train_chusai/A/*.tif"))
    image_B_paths, label_paths = [],[]
    for image_A_path in image_A_paths:
        image_B_path = image_A_path.replace("A","B")
        image_B_paths.append(image_B_path)
        label_path = image_A_path.replace("A","label").replace("tif","png")
        label_paths.append(label_path)
        
    train_loader = get_dataloader(image_A_paths, image_B_paths,
                                  label_paths, "train", batch_size,
                                  shuffle=True, num_workers=8)
    
    model_path = "Unet_resnet50a_dropout5.pth"
    # model_path = "../model/Unet_mobilenetv2.pth"
    
    train(epoches, batch_size, train_loader, model, optimizer, scheduler, 
          loss_fn, model_path)
    
    
    '''
    第二个模型
    '''
    
    # 定义模型
    model = Res_Unet(in_channels = channels, 
                      n_classes = 1, 
                      # backbone = 'resnet18_ibn_a',
                      backbone = "resnet34_ibn_a",
                      # backbone = "resnet50_ibn_a",
                      # model_path = 'resnet18_ibn_a-2f571257.pth', 
                      model_path = "resnet34_ibn_a-94bc1577.pth",
                      # model_path = "resnet50_ibn_a-d9d0bb7b.pth",
                      # backbone = 'resnet18_ibn_b',
                      # model_path = 'resnet18_ibn_b-bc2f3c11.pth', 
                      dropout_rate = 0.5,
                      scse = True, 
                      use_mish=False, 
                      db_block=False, 
                      hypercolumn=False)
    model.to(DEVICE)
    # model.load_state_dict(torch.load("../model/Unet_resnet50a_dropout5_45.pth"))

    # 采用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    
    scheduler = timm_scheduler.CosineLRScheduler(optimizer, 
                                                 t_initial=100,
                                                 lr_min=1e-5,
                                                 warmup_t=5,
                                                 warmup_lr_init=1e-5)
    
    loss_fn = edgeBCE_Dice_loss
    
    image_A_paths = sorted(glob.glob("/data/train_chusai/A/*.tif"))
    image_B_paths, label_paths = [],[]
    for image_A_path in image_A_paths:
        image_B_path = image_A_path.replace("A","B")
        image_B_paths.append(image_B_path)
        label_path = image_A_path.replace("A","label").replace("tif","png")
        label_paths.append(label_path)
        
    train_loader = get_dataloader(image_A_paths, image_B_paths,
                                  label_paths, "train", batch_size,
                                  shuffle=True, num_workers=8)
    
    model_path = "Unet_resnet34a_dropout5.pth"
    # model_path = "../model/Unet_mobilenetv2.pth"
    
    train(epoches, batch_size, train_loader, model, optimizer, scheduler, 
          loss_fn, model_path)



