# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:51:44 2021

@author: DELL
"""
import glob
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import time
import torch.utils.data as D
from torchvision import transforms as T
import json
import pycocotools.mask as mutils
import math
from scipy import ndimage
from model import Res_Unet

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 截断线性拉伸
def truncated_linear_stretch(image, truncated_value=0.2, max_out = 255, min_out = 0):
    image_stretch = []
    for i in range(image.shape[2]):
        gray = image[:,:,i]
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        image_stretch.append(gray)

    image_stretch = cv2.merge(image_stretch)
    image_stretch = np.uint8(image_stretch)
    return image_stretch

# 风格变换
def style_transfer(source_image, target_image, beta=0.005):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:,:,i])
        target_image_fshift = np.fft.fftshift(target_image_f)
        
        change_length = int(h*beta)
        source_image_fshift[int(h/2)-change_length:int(h/2)+change_length, 
                            int(h/2)-change_length:int(h/2)+change_length] = \
            target_image_fshift[int(h/2)-change_length:int(h/2)+change_length,
                                int(h/2)-change_length:int(h/2)+change_length]
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if>255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1,0).swapaxes(1,2)
    
    out = out.astype(np.uint8)
    return out

class OurDataset(D.Dataset):
    def __init__(self, image_A_paths, image_B_paths, label_paths, mode, test_json):
        self.image_A_paths = image_A_paths
        self.image_B_paths = image_B_paths
        self.label_paths = label_paths
        self.mode = mode
        self.test_json = test_json
        self.len = len(image_A_paths)
        
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
        
    # 获取数据操作
    def __getitem__(self, index):
        image_name = os.path.basename(self.image_A_paths[index])
        for test_info in self.test_json:
            if test_info['file_name'] == image_name:
                image_id = test_info["id"]
                break
        
        # 读取图像并转为RGB格式
        image_A = cv2.imread(self.image_A_paths[index],cv2.IMREAD_UNCHANGED)
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.imread(self.image_B_paths[index],cv2.IMREAD_UNCHANGED)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
        
        
        
        # 百分比截断拉伸
        image_A = truncated_linear_stretch(image_A)
        image_B = truncated_linear_stretch(image_B)
        
        # 风格变换
        # image_A = style_transfer(image_A, image_B)    # 掉点1.11
        # image_B = style_transfer(image_B, image_A)      # 掉点0.03
        
        
        h, w = image_A.shape[:2]
        padding = 200
        image_A_lt = image_A[:int(h/2)+padding,:int(w/2)+padding,:]
        image_A_lb = image_A[int(h/2)-padding:,:int(w/2)+padding,:]
        image_A_rt = image_A[:int(h/2)+padding,int(w/2)-padding:,:]
        image_A_rb = image_A[int(h/2)-padding:,int(w/2)-padding:,:]
        image_B_lt = image_B[:int(h/2)+padding,:int(w/2)+padding,:]
        image_B_lb = image_B[int(h/2)-padding:,:int(w/2)+padding,:]
        image_B_rt = image_B[:int(h/2)+padding,int(w/2)-padding:,:]
        image_B_rb = image_B[int(h/2)-padding:,int(w/2)-padding:,:]
        
        
        scale = 1.25
        zoom_hs, zoom_ws = [],[]
        
        h, w = image_A_lt.shape[:2]
        h_resize = math.ceil(h * scale / 32) * 32
        w_resize = math.ceil(w * scale / 32) * 32
        resize_shpae = (w_resize, h_resize)
        zoom_h = h / h_resize
        zoom_w = w / w_resize
        zoom_hs.append(zoom_h)
        zoom_ws.append(zoom_w)
        image_A_lt = cv2.resize(image_A_lt, resize_shpae)
        image_B_lt = cv2.resize(image_B_lt, resize_shpae)
        
        h, w = image_A_lb.shape[:2]
        h_resize = math.ceil(h * scale / 32) * 32
        w_resize = math.ceil(w * scale / 32) * 32
        resize_shpae = (w_resize, h_resize)
        zoom_h = h / h_resize
        zoom_w = w / w_resize
        zoom_hs.append(zoom_h)
        zoom_ws.append(zoom_w)
        image_A_lb = cv2.resize(image_A_lb, resize_shpae)
        image_B_lb = cv2.resize(image_B_lb, resize_shpae)
        
        h, w = image_A_rt.shape[:2]
        h_resize = math.ceil(h * scale / 32) * 32
        w_resize = math.ceil(w * scale / 32) * 32
        resize_shpae = (w_resize, h_resize)
        zoom_h = h / h_resize
        zoom_w = w / w_resize
        zoom_hs.append(zoom_h)
        zoom_ws.append(zoom_w)
        image_A_rt = cv2.resize(image_A_rt, resize_shpae)
        image_B_rt = cv2.resize(image_B_rt, resize_shpae)
        
        h, w = image_A_rb.shape[:2]
        h_resize = math.ceil(h * scale / 32) * 32
        w_resize = math.ceil(w * scale / 32) * 32
        resize_shpae = (w_resize, h_resize)
        zoom_h = h / h_resize
        zoom_w = w / w_resize
        zoom_hs.append(zoom_h)
        zoom_ws.append(zoom_w)
        image_A_rb = cv2.resize(image_A_rb, resize_shpae)
        image_B_rb = cv2.resize(image_B_rb, resize_shpae)
        
        

        image_A_B_lt = np.concatenate((image_A_lt, image_B_lt), axis=2)
        image_A_B_lb = np.concatenate((image_A_lb, image_B_lb), axis=2)
        image_A_B_rt = np.concatenate((image_A_rt, image_B_rt), axis=2)
        image_A_B_rb = np.concatenate((image_A_rb, image_B_rb), axis=2)
        
        return self.as_tensor(image_A_B_lt), self.as_tensor(image_A_B_lb), \
            self.as_tensor(image_A_B_rt), self.as_tensor(image_A_B_rb), \
                image_name, image_id, zoom_hs, zoom_ws
    
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_A_paths, image_B_paths, label_paths, mode, test_json, 
                   batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_A_paths, image_B_paths, label_paths, mode, test_json)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def test(models, model_paths, output_dir, test_loader, batch_size):

    dts = []
    for image_A_B_lt, image_A_B_lb, image_A_B_rt, image_A_B_rb, image_name, image_id, zoom_hs, zoom_ws in test_loader:
        
        output_lt, output_lb, output_rt, output_rb = 0,0,0,0
        
        for model, model_path in zip(models, model_paths):
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                image_A_B_lt = image_A_B_lt.cuda()
                image_A_B_lt_flip2 = torch.flip(image_A_B_lt,[2])
                image_A_B_lt_flip3 = torch.flip(image_A_B_lt,[3])
                output_lt1 = model(image_A_B_lt).cpu().data.numpy()
                output_lt2 = torch.flip(model(image_A_B_lt_flip2),[2]).cpu().data.numpy()
                output_lt3 = torch.flip(model(image_A_B_lt_flip3),[3]).cpu().data.numpy()
                
                image_A_B_lb = image_A_B_lb.cuda()
                image_A_B_lb_flip2 = torch.flip(image_A_B_lb,[2])
                image_A_B_lb_flip3 = torch.flip(image_A_B_lb,[3])
                output_lb1 = model(image_A_B_lb).cpu().data.numpy()
                output_lb2 = torch.flip(model(image_A_B_lb_flip2),[2]).cpu().data.numpy()
                output_lb3 = torch.flip(model(image_A_B_lb_flip3),[3]).cpu().data.numpy()
                
                image_A_B_rt = image_A_B_rt.cuda()
                image_A_B_rt_flip2 = torch.flip(image_A_B_rt,[2])
                image_A_B_rt_flip3 = torch.flip(image_A_B_rt,[3])
                output_rt1 = model(image_A_B_rt).cpu().data.numpy()
                output_rt2 = torch.flip(model(image_A_B_rt_flip2),[2]).cpu().data.numpy()
                output_rt3 = torch.flip(model(image_A_B_rt_flip3),[3]).cpu().data.numpy()
                
                image_A_B_rb = image_A_B_rb.cuda()
                image_A_B_rb_flip2 = torch.flip(image_A_B_rb,[2])
                image_A_B_rb_flip3 = torch.flip(image_A_B_rb,[3])
                output_rb1 = model(image_A_B_rb).cpu().data.numpy()
                output_rb2 = torch.flip(model(image_A_B_rb_flip2),[2]).cpu().data.numpy()
                output_rb3 = torch.flip(model(image_A_B_rb_flip3),[3]).cpu().data.numpy()
            
            
            output_lt += output_lt1 + output_lt2 + output_lt3
            output_lb += output_lb1 + output_lb2 + output_lb3
            output_rt += output_rt1 + output_rt2 + output_rt3
            output_rb += output_rb1 + output_rb2 + output_rb3
        
        # print(zoom_hs[0])
        # print(zoom_hs[1])
        # print(zoom_hs[0].numpy()[1])
        # print(zoom_hs[0][1].numpy())
        # print(zoom_hs,zoom_ws)
        output_lt = ndimage.interpolation.zoom(output_lt.squeeze(),zoom=(zoom_hs[0][0].numpy(), zoom_ws[0][0].numpy()),order=1)
        output_lb = ndimage.interpolation.zoom(output_lb.squeeze(),zoom=(zoom_hs[1][0].numpy(), zoom_ws[1][0].numpy()),order=1)
        output_rt = ndimage.interpolation.zoom(output_rt.squeeze(),zoom=(zoom_hs[2][0].numpy(), zoom_ws[2][0].numpy()),order=1)
        output_rb = ndimage.interpolation.zoom(output_rb.squeeze(),zoom=(zoom_hs[3][0].numpy(), zoom_ws[3][0].numpy()),order=1)
        
        output = np.concatenate((np.concatenate((output_lt[:-200,:],output_lb[200:,:]),0)[:,:-200],
                                 np.concatenate((output_rt[:-200,:],output_rb[200:,:]),0)[:,200:]),1)
        output = output / (len(model_paths)*3)
        
        pred = output
        threshold = 0.08
        mask = pred.copy()
        mask[pred>=threshold] = 1
        mask[pred<threshold] = 0
        mask = np.uint8(mask)

        nc, label = cv2.connectedComponents(mask, connectivity = 8)
        current_image_anns = []
        for c in range(nc):
            if np.all(mask[label == c] == 0):
                continue
            else:
                ann = np.asfortranarray((label == c).astype(np.uint8))
                current_image_anns.append(ann)
                rle = mutils.encode(ann)
                bbox = [int(_) for _ in mutils.toBbox(rle)]
                area = int(mutils.area(rle))
                score = float(pred[label == c].mean())
                if area>=10:
                    dts.append({
                        "segmentation": {
                            "size": [int(_) for _ in rle["size"]], 
                            "counts": rle["counts"].decode()},
                        "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                        "image_id": int(image_id[0]), "id": len(dts),
                        "score": float(score)
                    })
        
        save_path = os.path.join(output_dir, image_name[0].replace("tif","png"))
        print(save_path)
        cv2.imwrite(save_path, mask*255)
        cv2.imwrite(save_path+".png", np.uint8(pred*255))
        
        
        # # 双阈值-- 高阈值
        # threshold = 0.8
        # score = 0.2
        # mask = pred.copy()
        # mask[pred>=threshold] = 1
        # mask[pred<threshold] = 0
        # mask = np.uint8(mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # nc, label = cv2.connectedComponents(mask, connectivity = 8)
        # for c in range(nc):
        #     if np.all(mask[label == c] == 0):
        #         continue
        #     else:
        #         ann = np.asfortranarray((label == c).astype(np.uint8))
        #         rle = mutils.encode(ann)
        #         bbox = [int(_) for _ in mutils.toBbox(rle)]
        #         area = int(mutils.area(rle))
        #         # score = float(pred[label == c].mean())
        #         dts.append({
        #             "segmentation": {
        #                 "size": [int(_) for _ in rle["size"]], 
        #                 "counts": rle["counts"].decode()},
        #             "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
        #             "image_id": int(image_id[i]), "id": len(dts),
        #             "score": float(score)
        #         })
        
        # save_path = os.path.join(output_dir, image_name[i].replace(".tif","_.png"))
        # # print(save_path)
        # cv2.imwrite(save_path, mask*255)
            
            
            
    return dts
        
if __name__ == "__main__":
    start_time = time.time()
    in_channels = 6
    classes = 1
    model_34 = Res_Unet(in_channels = in_channels, 
                      n_classes = classes, 
                      backbone = "resnet34_ibn_a",
                      model_path = None,
                      dropout_rate = 0.5,
                      scse = True, 
                      use_mish=False, 
                      db_block=False, 
                      hypercolumn=False)
    
    model_50 = Res_Unet(in_channels = in_channels, 
                      n_classes = classes, 
                      backbone = "resnet50_ibn_a",
                      model_path = None,
                      dropout_rate = 0.5,
                      scse = True, 
                      use_mish=False, 
                      db_block=False, 
                      hypercolumn=False)
    
    models = [model_34, model_50]
    
    model_paths = [
        "../model/Unet_resnet34a_dropout5_44.pth",
        "../model/Unet_resnet50a_dropout5_44.pth",
        # "../model/Unet_resnet18a_dropout5_50.pth",
        ]
    
    test_json_path = r"E:\WangZhenQing\2022HongTu\project_building\data\changedetection_update\instances_test.json"
    
    with open(test_json_path, encoding="utf-8") as f:
        test_json = json.load(f)["images"]
        
        
    output_dir = r"E:\WangZhenQing\2022HongTu\project_building\data\changedetection_update"
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    
    image_A_paths = glob.glob('../data/changedetection_update/A/*.tif')
    image_B_paths = glob.glob("../data/changedetection_update/B/*.tif")
    batch_size = 1
    num_workers = 8
    test_loader = get_dataloader(image_A_paths, image_B_paths, None, "test", 
                                 test_json, batch_size, False, num_workers)
    dts = test(models, model_paths, output_dir, test_loader, batch_size)
    
    with open(r"E:\WangZhenQing\2022HongTu\project_building\data\results/test.segm.json", "w") as f:
        json.dump(dts, f)
    
    print((time.time()-start_time)/60**1)