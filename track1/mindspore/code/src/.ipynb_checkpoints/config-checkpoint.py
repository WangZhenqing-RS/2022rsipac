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

"""Configuration."""
import json
from easydict import EasyDict as ed


config_hrnetv2_w48 = ed({
    "data_url": None,
    "data_path": None,
    "train_url": None,
    "output_path": None,
    "checkpoint_url": None,
    "checkpoint_path": None,
    "eval_data_url": None,
    "eval_data_path": None,
    "run_distribute": False,
    "device_target": "Ascend",
    "workers": 1,
    "modelarts": True,
    "lr": 0.0001,
    # "lr_power": 4e-10,
    # "save_checkpoint_epochs": 40,
    # "keep_checkpoint_max": 20,
    "total_epoch": 100,
    "begin_epoch": 0,
    "end_epoch": 100,
    "batchsize": 8,
    "eval_callback": True,
    "eval_interval": 1,
    "seg_classes": 1,
    "det_classes":2,
    "train": ed({
        "train_list": "/train.lst",
        "image_size": [1024, 1024],
        "base_size": 1024,
        "multi_scale": False,
        "flip": False,
        "downsample_rate": 1,
        "scale_factor": 16,
        "shuffle": True,
        "param_initializer": "TruncatedNormal",
        "opt_momentum": 0.9,
        "wd": 0.0005,
        "num_samples": 0
    }),
    "dataset": ed({
        "name": "Cityscapes",
        "num_classes": 1,
        "ignore_label": -1,
        "mean": [0.390, 0.411, 0.361],
        "std": [0.203, 0.224, 0.225],

    }),
    "eval": ed({
        "eval_list": "/val.lst",
        "image_size": [1024, 1024],
        "base_size": 1024,
        "batch_size": 1,
        "num_samples": 0,
        "flip": False,
        "multi_scale": False,
        "scale_list": [1]
    }),
    "model": ed({
        "name": "seg_hrnet_w48",
        "num_outputs": 2,
        "align_corners": True,
        "extra": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [4],
                "NUM_CHANNELS": [64],
                "FUSE_METHOD": "SUM"
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [48, 96],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192, 384],
                "FUSE_METHOD": "SUM"
            },
        },
        "ocr": {
            "mid_channels": 512,
            "key_channels": 256,
            "dropout": 0.05,
            "scale": 1
        }
    }),
    "loss": ed({
        "loss_scale": 1024.0,
        "use_weights": True,
        "balance_weights": [0.4, 1]
    }),
})

config_hrnetv2_w18 = ed({
    "data_url": None,
    "data_path": None,
    "train_url": None,
    "output_path": None,
    "checkpoint_url": None,
    "checkpoint_path": None,
    "eval_data_url": None,
    "eval_data_path": None,
    "run_distribute": False,
    "device_target": "Ascend",
    "workers": 1,
    "modelarts": True,
    "lr": 0.001,
    "lr_power": 4e-10,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 10,
    "total_epoch": 100,
    "begin_epoch": 0,
    "end_epoch": 100,
    "batchsize": 8,
    "eval_callback": True,
    "eval_interval": 1,
    "seg_classes": 1,
    "det_classes":2,
    "train": ed({
        "train_list": "/train.lst",
        "image_size": [1024, 1024],
        "base_size": 1024,
        "multi_scale": False,
        "flip": False,
        "downsample_rate": 1,
        "scale_factor": 16,
        "shuffle": True,
        "param_initializer": "TruncatedNormal",
        "opt_momentum": 0.9,
        "wd": 0.0005,
        "num_samples": 0
    }),
    "dataset": ed({
        "name": "Cityscapes",
        "num_classes": 1,
        "ignore_label": -1,
        "mean": [0.390, 0.411, 0.361],
        "std": [0.203, 0.224, 0.225],

    }),
    "eval": ed({
        "eval_list": "/val.lst",
        "image_size": [1024, 1024],
        "base_size": 1024,
        "batch_size": 1,
        "num_samples": 0,
        "flip": False,
        "multi_scale": False,
        "scale_list": [1]
    }),
    "model": ed({
        "name": "seg_hrnet_w18",
        "num_outputs": 2,
        "align_corners": True,
        "extra": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [4],
                "NUM_CHANNELS": [64],
                "FUSE_METHOD": "SUM"
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [18, 36],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [18, 36, 72],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [18, 36, 72, 144],
                "FUSE_METHOD": "SUM"
            },
        },
        "ocr": {
            "mid_channels": 512,
            "key_channels": 256,
            "dropout": 0.05,
            "scale": 1
        }
    }),
    "loss": ed({
        "loss_scale": 1024.0,
        "use_weights": True,
        "balance_weights": [0.4, 1]
    }),
})

def show_config(cfg):
    """Show configuration."""
    split_line_up = "==================================================\n"
    split_line_bt = "\n=================================================="
    print(split_line_up,
          json.dumps(cfg, ensure_ascii=False, indent=2),
          split_line_bt)


def organize_configuration(cfg, args):
    """Add parameters from command-line into configuration."""
    args_dict = vars(args)
    for item in args_dict.items():
        cfg[item[0]] = item[1]
