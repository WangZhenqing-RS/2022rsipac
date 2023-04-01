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
        
        
        # # 去除全零行
        # image_A = image_A[[not np.all(image_A[i] == 0) for i in range(image_A.shape[0])], :]
        # # 去除全零列
        # image_A = image_A[:, [not np.all(image_A[:, i] == 0) for i in range(image_A.shape[1])]]
        
        # image_B = image_B[[not np.all(image_B[i] == 0) for i in range(image_B.shape[0])], :]
        # image_B = image_B[:, [not np.all(image_B[:, i] == 0) for i in range(image_B.shape[1])]]
        
        
        # # 从上往下全零值的行数
        # zero_row_start = 0
        # # 从下往上全零值的行数
        # zero_row_end = 0
        # # 从左往右全零值的列数
        # zero_col_start = 0
        # # 从右往左全零值的列数
        # zero_col_end = 0
        # for i in range(image_A.shape[0]):
        #     row = i
        #     if np.all(image_A[row] == 0):
        #         zero_row_start = i+1
        #     else:
        #         break
        # for i in range(image_A.shape[0]):
        #     row = image_A.shape[0]-1-i
        #     if np.all(image_A[row] == 0):
        #         zero_row_end = i+1
        #     else:
        #         break
        # for i in range(image_A.shape[1]):
        #     col = i
        #     if np.all(image_A[:,col] == 0):
        #         zero_col_start = i+1
        #     else:
        #         break
        # for i in range(image_A.shape[1]):
        #     col = image_A.shape[1]-1-i
        #     if np.all(image_A[:,col] == 0):
        #         zero_col_end = i+1
        #     else:
        #         break
        
        # image_A = image_A[zero_row_start:-zero_row_end if zero_row_end!=0 else None,
        #                   zero_col_start:-zero_col_end if zero_col_end!=0 else None,]
        # image_B = image_B[zero_row_start:-zero_row_end if zero_row_end!=0 else None,
        #                   zero_col_start:-zero_col_end if zero_col_end!=0 else None,]
        
        
        # 百分比截断拉伸
        image_A = truncated_linear_stretch(image_A)
        image_B = truncated_linear_stretch(image_B)
        
        if image_name in ["5.tif","6.tif","7.tif"]:
            image_A_padding = np.zeros((image_A.shape[0]+400,image_A.shape[1]+400,3),np.uint8)
            # print(image_A_padding.shape, image_A_padding[200:-200,200:-200].shape)
            image_A_padding[200:-200,200:-200] = image_A
            image_A = image_A_padding
            image_B_padding = np.zeros((image_B.shape[0]+400,image_B.shape[1]+400,3),np.uint8)
            image_B_padding[200:-200,200:-200] = image_B
            image_B = image_B_padding
            
        
        # 风格变换
        # image_A = style_transfer(image_A, image_B)    # 掉点1.11
        # image_B = style_transfer(image_B, image_A)      # 掉点0.03
        
        h, w = image_A.shape[:2]
        # original_shpae = (w, h)
        print("ori: ", h, w)
        
        scale = 0.75
        
        h_resize = math.ceil(h * scale / 32) * 32
        w_resize = math.ceil(w * scale / 32) * 32
        resize_shpae = (w_resize, h_resize)
        print("res: ", h_resize, w_resize)
        
        zoom_h = h / h_resize
        zoom_w = w / w_resize
        
        print("zoom: ", zoom_h, zoom_w)
        image_A = cv2.resize(image_A, resize_shpae)
        image_B = cv2.resize(image_B, resize_shpae)
        
        # A_save_path = r"E:\WangZhenQing\2022HongTu\project_building\data\changedetection_update\Dataset_A_str/" + image_name
        # cv2.imwrite(A_save_path,image_A)
        # B_save_path = A_save_path.replace("A_str","B_str")
        # cv2.imwrite(B_save_path,image_B)

        image_A_B = np.concatenate((image_A, image_B), axis=2)
        
        # return self.as_tensor(image_A_B), image_name, image_id, zoom_h, zoom_w, zero_row_start, zero_row_end, zero_col_start, zero_col_end
        return self.as_tensor(image_A_B), image_name, image_id, zoom_h, zoom_w
    
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
    # for image_A_B, image_name, image_id, zoom_h, zoom_w, zero_row_start, zero_row_end, zero_col_start, zero_col_end in test_loader:
    for image_A_B, image_name, image_id, zoom_h, zoom_w in test_loader:
        
        output = 0
        
        for model, model_path in zip(models, model_paths):
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                # image.shape: batch_size,3,512,512
                image_A_B = image_A_B.cuda()
                image_A_B_flip2 = torch.flip(image_A_B, [2])
                image_A_B_flip3 = torch.flip(image_A_B, [3])
                
                # image_A_B_rot90 = torch.rot90(image_A_B, 1, [2, 3])
                
                
                output1 = model(image_A_B).cpu().data.numpy()
                output2 = torch.flip(model(image_A_B_flip2), [2]).cpu().data.numpy()
                output3 = torch.flip(model(image_A_B_flip3), [3]).cpu().data.numpy()
                
                # output4 = torch.rot90(model(image_A_B_rot90), -1, [2, 3]).cpu().data.numpy()
            
            output += output1 + output2 + output3
            # output += output1
        
        output = output / (len(model_paths)*3)
        # output.shape: batch_size,classes,512,512
        for i in range(output.shape[0]):
            pred = output[i]
            pred = pred.squeeze()
            
            # print(zoom_h.numpy(), zoom_w.numpy())
            # print(zoom_h.numpy()[i], zoom_w.numpy()[i])
            # print(zoom_h[i],zoom_w[i])
            
            pred = ndimage.interpolation.zoom(pred,zoom=(zoom_h.numpy()[i], zoom_w.numpy()[i]),order=1)
            # pred = cv2.resize(pred, original_shpae[i])
            
            # print(pred.shape)
            # print(zero_row_start, zero_row_end, zero_col_start, zero_col_end)
            # if zero_row_start==0 and zero_row_end==0 and zero_col_start==0 and zero_col_end==0:
            #     pass
            # else:
            #     pred_ture = np.zeros((pred.shape[0]+zero_row_start+zero_row_end, pred.shape[1]+zero_col_start+zero_col_end))
            #     pred_ture[zero_row_start:-zero_row_end if zero_row_end!=0 else None,
            #               zero_col_start:-zero_col_end if zero_col_end!=0 else None] = pred
            #     pred = pred_ture
            
            
            if image_name[i] in ["5.tif","6.tif","7.tif"]:
                pred = pred[200:-200,200:-200]
            
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
                            "image_id": int(image_id[i]), "id": len(dts),
                            "score": float(score)
                        })
            
            save_path = os.path.join(output_dir, image_name[i].replace("tif","png"))
            print(save_path)
            cv2.imwrite(save_path, mask*255)
            cv2.imwrite(save_path+".png", np.uint8(pred*255))
            
            
            # # 双阈值-- 高阈值
            # threshold = 0.65
            # score = 0.12
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