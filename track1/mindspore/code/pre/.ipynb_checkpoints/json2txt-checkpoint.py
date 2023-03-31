# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:14:23 2022

@author: DELL
"""
import os
import json
import tqdm
import shutil

def get_img_annos(img_id, annos):
    
    anno_list = []
    for anno_info in annos["annotations"]:
        if anno_info['image_id'] == img_id:
            anno_bbox = anno_info["bbox"]
            xmin = anno_bbox[0]
            ymin = anno_bbox[1]
            w = anno_bbox[2]
            h = anno_bbox[3]
            xmax = xmin + w
            ymax = ymin + h
            # 1,2 -> 0,1
            category_id = anno_info["category_id"]-1
            anno_list.append([xmin,ymin,xmax,ymax,category_id])
        else:
            continue
    return anno_list 


def json2txt(json_path, save_folder):

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    
    with open(json_path, encoding="utf-8") as f:
        annotations_train = json.load(f)
        
    for img_info in tqdm.tqdm(annotations_train["images"]):
        image_name = img_info['file_name']
        image_id = img_info['id']
        image_annos = get_img_annos(image_id, annotations_train)
        
        image_anno_path = os.path.join(save_folder,image_name.replace("tif","txt"))
        with open(image_anno_path, 'a') as f:
            for image_anno in image_annos:
                f.write(f"{image_anno[0]} {image_anno[1]} {image_anno[2]} {image_anno[3]} {image_anno[4]}\n")
            
        # break
    
if __name__ == '__main__':
    json_path = "../../data/fusai_release/train/instances_train.json"
    save_folder = "../../data/fusai_release/train/annotations"
    json2txt(json_path, save_folder)