import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.hrnet import BN_MOMENTUM, hrnet_classification


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class HRnet(nn.Module):
    def __init__(self, seg_classes = 1, det_classes = 2, backbone = 'hrnetv2_w18', pretrained = True):
        super(HRnet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)
        
        in_channels = np.int(np.sum(self.backbone.model.pre_stage_channels))
        inter_channels = in_channels // 4
        
        dropout_pro = 0.1
        
        # 分割部分
        self.fcn_head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_pro, False),
            nn.Conv2d(in_channels=inter_channels, out_channels=seg_classes, kernel_size=1, stride=1, padding=0)
        )
        
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_pro, False),
            nn.Conv2d(inter_channels, det_classes, kernel_size=1, stride=1, padding=0))
        
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_pro, False),
            nn.Conv2d(inter_channels, 2, kernel_size=1, stride=1, padding=0))
        

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_pro, False),
            nn.Conv2d(inter_channels, 2, kernel_size=1, stride=1, padding=0))



    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)
        
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        
        # print(x[3].shape)
        
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        
        
        x = torch.cat([x[0], x1, x2, x3], 1)
        
        seg = self.fcn_head(x).sigmoid_()
        
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)
        
        
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        
        return seg, hm, wh, offset
