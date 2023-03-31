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

"""OCRNet training."""
import os
import ast
import math
import argparse
import numpy as np
import random
import warnings
import mindspore

import mindspore.dataset as de
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore import context, Model
from mindspore.nn import SGD
import mindspore.nn as nn 
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig

# from src.config import config_hrnetv2_w48 as config
from src.config import config_hrnetv2_w18 as config

from src.utils import Train_checkpoint_save_moxing, StepLossTimeMonitor, PolyLR
from src.config import organize_configuration
from src.road_dataset import RoadDataset
from src.hrnet import get_seg_model
from src.loss import RoadLoss
from src.callback import EvalCallback
from src.model_utils.moxing_adapter import moxing_wrapper

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置随机种子
def seed_it(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)
    mindspore.set_seed(seed)
    mindspore.common.set_seed(seed)
    de.config.set_seed(seed)

# def get_cos_lr(base_lr, steps, warmup_steps, warm_start_lr):
#     """Get learning rates for each step."""
#     lrs = []
#     for step in steps:
#         # 线性变换 warm up
#         if step < warmup_steps: 
#             lr = 1.0 * step / warmup_steps * base_lr + warm_start_lr
#             lrs.append(lr)
#         # 余弦退火 cos
#         else:
#             lr = 0.5 * base_lr * (1 + math.cos(1.0 * (step-warmup_steps) / (len(steps)-warmup_steps) * math.pi))
#             lrs.append(lr)
#     return lrs

# def get_exp_lr(base_lr, xs, power=4e-10):
#     """Get learning rates for each step."""
#     ys = []
#     for x in xs:
#         ys.append(base_lr / np.exp(power*x**2))
#     return ys

class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, image, label, hm, wh, reg, reg_mask):
        output = self._backbone(image)
        return self._loss_fn(output, label, hm, wh, reg, reg_mask)
    
def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore OCRNet Training Configurations.")
    parser.add_argument("--data_url", type=str, default=None, help="Storage path of dataset in OBS.")
    parser.add_argument("--train_url", type=str, default=None, help="Storage path of training results in OBS.")
    parser.add_argument("--data_path", type=str, default="./data/train", help="Storage path of dataset on machine.")
    parser.add_argument("--output_path", type=str, default="./model", help="Storage path of training results on machine.")
    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming on machine.")
    parser.add_argument("--workers", type=int, default=8,
                        help="The number of workers in the data iterator.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=True,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Base learning rate.")
    parser.add_argument("--begin_epoch", type=int, default=0,
                        help="If it's a training resuming task, give it a beginning epoch.")
    parser.add_argument("--end_epoch", type=int, default=40,
                        help="If you want to stop the task early, give it an ending epoch.")
    parser.add_argument("--total_epoch", type=int, default=100,
                        help="total_epoch.")
    parser.add_argument("--batchsize", type=int, default=6,
                        help="batch size.")
    parser.add_argument("--eval_callback", type=ast.literal_eval, default=False,
                        help="To use inference while training or not.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Epoch interval of evaluating result during training.")
    return parser.parse_args()


@moxing_wrapper(config)
def main():
    """Training process."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    # context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    if config.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 0
        device_num = 1

    # Create dataset
    train_annotation_path   = "code_mindspore/train_all.txt"
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
        
    # train_lines = train_lines[:16]
    input_shape = (1024,1024)
    seg_classes = 1
    det_classes = 2
    
    data_tr = RoadDataset(train_lines, 
                          input_shape, 
                          seg_classes, 
                          det_classes)
    
    # dataset.show()
    if device_num == 1:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label", "hm", "wh", "reg", "reg_mask"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.train.shuffle)
    else:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label", "hm", "wh", "reg", "reg_mask"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.train.shuffle,
                                      num_shards=device_num, shard_id=device_id)
    dataset = dataset.batch(config.batchsize, drop_remainder=True)
    
    data_size = dataset.get_dataset_size()
    print("dataset length is:", data_size)
    
    # 创建网络
    net = get_seg_model(config)
    local_checkpoint_url = "./pre_model/hrnetv2_w18_imagenet_pretrained.ckpt"
    pretrained_dict = load_checkpoint(local_checkpoint_url)
    # filter_list = [x.name for x in net.last_layer.get_parameters()]
    # filter_checkpoint_parameter_by_list(pretrained_dict, filter_list)
    param_not_load = load_param_into_net(net, pretrained_dict)
    # print(param_not_load)
    net.set_train(True)

    # 创建损失函数
    loss = RoadLoss()
    loss_scale_manager = FixedLossScaleManager(config.loss.loss_scale, False)
    
    # # 设置学习率调整策略
    # steps_per_epoch = dataset.get_dataset_size()
    # total_steps = config.total_epoch * steps_per_epoch
    # begin_step = config.begin_epoch * steps_per_epoch
    # end_step = config.end_epoch * steps_per_epoch
    # xs = np.linspace(0, total_steps, total_steps)
    # warmup_steps = 5 * steps_per_epoch
    # warm_start_lr = 1e-5
    # lr = get_cos_lr(config.lr, xs, warmup_steps, warm_start_lr)
    # lr = lr[begin_step: end_step]
    
    opt = nn.Adam(params=net.trainable_params(), learning_rate=config.lr, weight_decay=0.0001,
                  loss_scale=config.loss.loss_scale)
    
    # 定义损失网络，连接前向网络和多标签损失函数
    loss_net = NetWithLoss(net, loss)
    
    # 创建模型
    # 在Ascend上建议使用”O3”，将网络精度（包括BatchNorm）转为float16，不使用损失缩放策略。
    model = Model(network=loss_net, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O3",
                  keep_batchnorm_fp32=False)
    
    # 回调函数
    # # 表示每隔多少个step打印一次时间信息
    # time_cb = TimeMonitor(data_size=steps_per_epoch)
    # # 表示每隔多少个step打印一次loss
    # loss_cb = LossMonitor(per_print_times=1)
    # # 表示每隔多少个step保存模型
    # ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * config.save_checkpoint_epochs,
    #                                keep_checkpoint_max=config.keep_checkpoint_max)
    # # 模型保存地址和后缀
    # ckpt_cb = ModelCheckpoint(prefix='{}'.format("hrnet48"),
    #                           directory=config.output_path+"/card" + str(device_id),
    #                           config=ckpt_config)
    
    ckpoint_cb = Train_checkpoint_save_moxing(config.output_path, "obs://demo/")
    cb = [
        StepLossTimeMonitor(batch_size=config.batchsize, per_print_times=1, train_data_size=data_size),
        ckpoint_cb,
        PolyLR(config.lr,100)
        # time_cb, 
        # loss_cb,
        # ckpt_cb
         ]

    train_epoch = config.end_epoch - config.begin_epoch
    model.train(train_epoch, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    seed_it(42)
    args = parse_args()
    organize_configuration(cfg=config, args=args)
    main()
