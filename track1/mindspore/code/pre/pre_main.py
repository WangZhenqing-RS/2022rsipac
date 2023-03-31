# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:36:34 2022

@author: DELL
"""

import argparse

from json2txt import json2txt
from data_clip import data_clip
from gen_train_txt import gen_train_txt



def parse_args():
    """
    Get arguments from command-line.
    """
    parser = argparse.ArgumentParser(description='data preprocess.')
    parser.add_argument("--chusai_train_data_dir", type=str, default="././data/chusai_release/train")
    parser.add_argument("--fusai_train_data_dir", type=str, default="././data/fusai_release/train")
    parser.add_argument("--train_all_txt_path", type=str, default="code_mindspore/train_all.txt")
    
    return parser.parse_args(args=[])

if __name__ == '__main__':
    
    args = parse_args()
    chusai_train_data_dir = args.chusai_train_data_dir
    fusai_train_data_dir = args.fusai_train_data_dir
    train_all_txt_path = args.train_all_txt_path
    
    chusai_json_path = chusai_train_data_dir + "/instances_train.json"
    chusai_ann_folder = chusai_train_data_dir + "/annotations"
    chusai_img_floder = chusai_train_data_dir +"/images"
    chusai_label_folder = chusai_train_data_dir + "/masks"
    
    chusai_out_img_floder = chusai_train_data_dir + "/images_crop"
    chusai_out_label_floder = chusai_train_data_dir + "/masks_crop"
    chusai_out_anno_floder = chusai_train_data_dir + "/annotations_crop"
    
    fusai_json_path = fusai_train_data_dir + "/instances_train.json"
    fusai_ann_folder = fusai_train_data_dir + "/annotations"
    fusai_img_floder = fusai_train_data_dir +"/images"
    fusai_label_folder = fusai_train_data_dir + "/masks"
    
    fusai_out_img_floder = fusai_train_data_dir + "/images_crop"
    fusai_out_label_floder = fusai_train_data_dir + "/masks_crop"
    fusai_out_anno_floder = fusai_train_data_dir + "/annotations_crop"
    
    
    print("step 1: chusai json to txt...")
    json2txt(chusai_json_path, chusai_ann_folder)
    
    print("step 2: chusai data clip...")
    data_clip(chusai_json_path, chusai_img_floder, chusai_label_folder, chusai_ann_folder, 
              chusai_out_img_floder, chusai_out_label_floder, chusai_out_anno_floder)
    
    print("step 3: fusai json to txt...")
    json2txt(fusai_json_path, fusai_ann_folder)
    
    print("step 4: fusai data clip...")
    data_clip(fusai_json_path, fusai_img_floder, fusai_label_folder, fusai_ann_folder,
              fusai_out_img_floder, fusai_out_label_floder, fusai_out_anno_floder)
    
    print("step 5: gen train txt...")
    gen_train_txt(chusai_out_anno_floder, fusai_out_anno_floder, train_all_txt_path)