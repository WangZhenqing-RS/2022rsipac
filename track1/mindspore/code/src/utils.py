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

"""Initialize bias of convolution cell."""
import os
import time
import math
import moxing as mox
import numpy as np
import mindspore
from mindspore.common import initializer
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
from mindspore import save_checkpoint

class PolyLR(Callback):
    def __init__(self, lr, epochs=100, min_lr=0):
        super(PolyLR, self).__init__()
        self.epochs = epochs
        self.orglr = lr

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        new_lr = self.orglr*(1-(float(cur_epoch)/self.epochs)**0.9)
        mindspore.ops.functional.assign(cb_params['optimizer'].learning_rate, Tensor(new_lr, mindspore.float32))
        
class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: False.

    Returns:
        Tensor, feature at spectified position
    """
    def __init__(self, enable_cpu_gather=False):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.enable_cpu_gather = enable_cpu_gather
        if self.enable_cpu_gather:
            self.gather_nd = ops.GatherD()
            self.expand_dims = ops.ExpandDims()
        else:
            self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by specified index"""
        if self.enable_cpu_gather:
            _, _, c = self.shape(feat)
            # (b, N, c)
            index = self.expand_dims(ind, -1)
            index = self.tile(index, (1, 1, c))
            feat = self.gather_nd(feat, 1, index)
        else:
            # (b, N)->(b*N, 1)
            b, N = self.shape(ind)
            ind = self.reshape(ind, (-1, 1))
            ind_b = nn.Range(0, b, 1)()
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.tile(ind_b, (1, N))
            ind_b = self.reshape(ind_b, (-1, 1))
            index = self.concat((ind_b, ind))
            # (b, N, 2)
            index = self.reshape(index, (b, N, -1))
            # (b, N, c)
            feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """
    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        # (b, c, h, w)->(b, h*w, c)
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat
    

def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def get_conv_bias(cell):
    weight = initializer.initializer(initializer.HeUniform(negative_slope=math.sqrt(5)),
                                     cell.weight.shape, cell.weight.dtype).to_tensor()
    fan_in, _ = calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    return initializer.initializer(initializer.Uniform(scale=bound),
                                   cell.bias.shape, cell.bias.dtype)


class StepLossTimeMonitor(Callback):

    def __init__(self, batch_size, per_print_times=1, train_data_size=0):
        super(StepLossTimeMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.batch_size = batch_size
        self.train_data_size = train_data_size


    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):

        step_seconds = time.time() - self.step_time
        step_fps = self.batch_size * 1.0 / step_seconds

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        self.losses.append(loss)


    def epoch_begin(self, run_context):
        self.epoch_start = time.time()
        self.losses = []
        cb_params = run_context.original_args()
        print("")
        print("epoch: {:3d}, start training".format(cb_params.cur_epoch_num), flush=True)
        # print("epoch: {:3d}, lr:{:.8f}".format(cb_params.cur_epoch_num, cb_params['optimizer'].get_lr().asnumpy()[0]), flush=True)


    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_cost = time.time() - self.epoch_start
        step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        step_fps = self.batch_size * 1.0 * step_in_epoch / epoch_cost
        print("")
        print("epoch: {:3d}, avg loss:{:.4f}, total cost: {:.3f} s, per step fps:{:5.3f}".format(
            cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost, step_fps), flush=True)
        print("==============EPOCH END==============")
        
        
class Train_checkpoint_save_moxing(Callback):
    def __init__(self, loacldir, obs_dir = ''):
        super(Train_checkpoint_save_moxing, self).__init__()
        if not mox.file.is_directory(obs_dir):
            mox.file.make_dirs(obs_dir)
        if os.path.exists(loacldir) is False:
            os.makedirs(loacldir)
            
        self.ckpt_directory = loacldir
        self.obs_directory = obs_dir

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch>15:
            ckpt_path = os.path.join(self.ckpt_directory, "epoch_{}.ckpt".format(cur_epoch))
            save_checkpoint(cb_params.train_network, ckpt_path)
            
            # 有时会报错
            # ERROR:root:Failed to call:
            #     func=<bound method ObsClient.putContent of <moxing.framework.file.src.obs.client.ObsClient object at 0xfffef8c137d0>>
            #     args=('demo', '/')
            #     kwargs={content:None, headers:{'acl': 'bucket-owne, }
            # ERROR:root:
            #     stat:403
            #     errorCode:AccessDenied
            #     errorMessage:Access Denied
            #     reason:Forbidden
            #     request-id:000001843F7225C08144ED3A9E238F1A
            #     retry:0
            
            # # 移动到桶
            # mox.file.copy_parallel(self.ckpt_directory, self.obs_directory)
            
            print('============== save to ', ckpt_path)
