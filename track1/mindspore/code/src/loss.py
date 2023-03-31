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

"""Loss functions."""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
import mindspore.ops as F
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
# from mindspore.nn.loss.loss import LossBase

from src.config import config_hrnetv2_w48 as config


weights_list = [1.0, 1.2]



class DiceLoss(nn.Cell):
    def __init__(self, isact=True):
        super(DiceLoss, self).__init__()
        self.isact = isact
        self.act = mindspore.ops.Sigmoid()
        self.eps = 1e-7
        self.reduce_sum = F.ReduceSum()

    def construct(self, logits, label):
        if self.isact:
            logits = self.act(logits)
        
        # float16求和会出问题
        logits = ops.Cast()(logits, mindspore.dtype.float32)
        label = ops.Cast()(label, mindspore.dtype.float32)
        
        tp = self.reduce_sum(logits*label)
        fp = self.reduce_sum(logits)
        fn = self.reduce_sum(label)
        
        # print(tp, fp, fn)
        score = (2*tp + self.eps)/(fn + fp + self.eps)
        return 1 - score
    

class FocalLoss(nn.Cell):
    """
    Warpper for focal loss.

    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.

    Returns:
        Tensor, focal loss.
    """
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()
        self.transpose = ops.Transpose()

    def construct(self, out, target):
        out = self.transpose(out,(0,2,3,1))
        """focal loss"""
        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        neg_inds = self.cast(self.less(target, 1.0), mstype.float32)
        neg_weights = self.pow(1 - target, self.beta)

        pos_loss = self.log(out) * self.pow(1 - out, self.alpha) * pos_inds
        neg_loss = self.log(1 - out) * self.pow(out, self.alpha) * neg_weights * neg_inds

        num_pos = self.reduce_sum(pos_inds, ())
        num_pos = self.select(self.equal(num_pos, 0.0),
                              self.fill(self.dtype(num_pos), self.shape(num_pos), 1.0), num_pos)
        pos_loss = self.reduce_sum(pos_loss, ())
        neg_loss = self.reduce_sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss


class RegLoss(nn.Cell): #reg_l1_loss
    """
    Warpper for regression loss.

    Args:
        mode(str): L1 or Smoothed L1 loss. Default: "l1"

    Returns:
        Tensor, regression loss.
    """
    def __init__(self, mode='l1'):
        super(RegLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        if mode == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        elif mode == 'sl1':
            self.loss = nn.SmoothL1Loss()
        else:
            self.loss = None

    def construct(self, output, target, mask):
        pred = self.transpose(output,(0,2,3,1))
        mask = self.cast(mask, mstype.float32)
        # print(output.shape, pred.shape,target.shape, mask.shape)
        num = self.reduce_sum(mask, ())
        mask = self.expand_dims(mask, 3)
        # print(target.shape, mask.shape)
        target = target * mask
        pred = pred * mask
        regr_loss = self.loss(pred, target)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss

class RoadLoss(nn.Cell):

    def __init__(self):
        super(RoadLoss, self).__init__()
        self.weights = config.loss.balance_weights
        self.align_corners = config.model.align_corners
        self.resize_bilinear = nn.ResizeBilinear()
        self.act = ops.Sigmoid()
        self.bceLoss = nn.BCELoss(reduction='mean')
        self.diceLoss = DiceLoss(isact=False)
        # self.diceLoss = nn.DiceLoss(smooth=1e-5)
        self.clsLoss = FocalLoss()
        self.regLoss = RegLoss()

    def construct(self, score, target_label, target_hm, target_wh, target_reg, target_reg_mask):
        
        seg, hm, wh, offset = score
        
        # seg = ops.Cast()(seg, mindspore.dtype.float32)
        # hm = ops.Cast()(hm, mindspore.dtype.float32)
        # wh = ops.Cast()(wh, mindspore.dtype.float32)
        # offset = ops.Cast()(offset, mindspore.dtype.float32)
        # target_label = ops.Cast()(target_label, mindspore.dtype.float32)
        # target_hm = ops.Cast()(target_hm, mindspore.dtype.float32)
        # target_wh = ops.Cast()(target_wh, mindspore.dtype.float32)
        # target_reg = ops.Cast()(target_reg, mindspore.dtype.float32)
        # target_reg_mask = ops.Cast()(target_reg_mask, mindspore.dtype.float32)
        
        
        ph, pw = seg.shape[2], seg.shape[3]
        h, w = target_label.shape[2], target_label.shape[3]
        if ph != h or pw != w:
            seg = self.resize_bilinear(seg, size=(h, w), align_corners=self.align_corners)
            
        seg = self.act(seg)
        hm  = self.act(hm)
        
        target_label = ops.Cast()(target_label, mindspore.dtype.float32)
        
        bce_loss = self.bceLoss(seg, target_label)
        dice_loss = self.diceLoss(seg, target_label)
        
        cls_loss = self.clsLoss(hm, target_hm)
        wh_loss = self.regLoss(wh, target_wh, target_reg_mask)
        off_loss = self.regLoss(offset, target_reg, target_reg_mask)
        # print(f"bce_loss: {bce_loss}; dice_loss: {dice_loss}; cls_loss: {cls_loss}; wh_loss: {wh_loss}; off_loss: {off_loss}")
        # return bce_loss + dice_loss
        return bce_loss + dice_loss + 0.1*(cls_loss + 0.1*wh_loss + off_loss)