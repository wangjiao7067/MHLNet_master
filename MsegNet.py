#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/30 21:25
# @Author  : Star
# @File    : mda_seunet_aux.py
# @Software: PyCharm
from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
from SlocNet import RoiSlocNet

class BranchAtten(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, split_dimen=1, permute1=None):
        super(BranchAtten, self).__init__()
        if permute1 is None:
            self.permute1 = [0, 1, 2, 3, 4]
        else:
            self.permute1 = permute1

        self.split_dimen = split_dimen
        self.conv1 = nn.Conv3d(in_channels1, out_channels, 1, 1, 0)
        self.conv2 = nn.Conv3d(in_channels2, 1, 1, 1, 0)
        self.last_conv = nn.Conv3d(in_channels1, out_channels, 1, 1, 0)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        if self.split_dimen != 1:
            x_reshape1 = x_conv1.permute(self.permute1[0], self.permute1[1], self.permute1[2],
                                         self.permute1[3], self.permute1[4])
            x_conv2 = self.conv2(x_reshape1)
        else:
            x_conv2 = self.conv2(x_conv1)
        x_para = torch.sigmoid(x_conv2)

        x = x.permute(self.permute1[0], self.permute1[1], self.permute1[2], self.permute1[3], self.permute1[4])

        x_out = x * x_para
        x_out = x_out.permute(self.permute1[0], self.permute1[1], self.permute1[2], self.permute1[3], self.permute1[4])
        return x_out


class MDCA(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, shape=None):
        super(MDCA, self).__init__()
        self.branch1 = BranchAtten(in_channels, inter_channels, out_channels, 1, [0, 1, 2, 3, 4])
        self.branch2 = BranchAtten(in_channels, shape[0], out_channels, 2, [0, 2, 1, 3, 4])
        self.branch3 = BranchAtten(in_channels, shape[1], out_channels, 3, [0, 3, 2, 1, 4])
        self.branch4 = BranchAtten(in_channels, shape[2], out_channels, 4, [0, 4, 2, 3, 1])
        self.channel_atten = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x_branch1 = self.branch1(x)
        x_branch2 = self.branch2(x)
        x_branch3 = self.branch3(x)
        x_branch4 = self.branch4(x)
        x_channel = self.channel_atten(x)
        x_channel = torch.sigmoid(x_channel)
        x_channel = x * x_channel
        x_out = x + x_branch1 + x_branch2 + x_branch3 + x_branch4 + x_channel
        return x_out

class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, 3, 1, 1)
        self.sn1 = FastSmoothSENorm(out_channels // 2, 2)
        self.sn2 = FastSmoothSENorm(out_channels, 2)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.relu(x)
        return x, self.pool(x)


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv3d(in_channels // 2 + in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.sn1 = FastSmoothSENorm(out_channels, 2)
        self.sn2 = FastSmoothSENorm(out_channels, 2)
        self.relu = nn.ReLU(True)
        self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        x = cat((x, x1), dim=1)
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.relu(x)
        return x


class MsegNet(nn.Module):
    def __init__(self, ):
        super(MsegNet, self).__init__()
        init_channels = 1
        class_nums = 9
        batch_norm = True
        sample = True

        self.en1 = Downsample_block(init_channels, 24)
        self.en2 = Downsample_block(24, 48)
        self.en3 = Downsample_block(48, 96)
        self.en4 = Downsample_block(96, 192)

        self.up3 = Upsample_block(192, 96)
        self.multi_dimen_atten3 = MDCA(96, 96, 96, shape=[18, 48, 48])
        self.up2 = Upsample_block(96, 48)
        self.multi_dimen_atten2 = MDCA(48, 48, 48, shape=[36, 96, 96])
        self.up1 = Upsample_block(48, 24)
        self.multi_dimen_atten1 = MDCA(24, 24, 24, shape=[72, 192, 192])
        self.con_last = nn.Conv3d(168, class_nums, 1)

        self.sample = nn.ConvTranspose3d(96, 48, 2, stride=2)

        self.slocnet = RoiSlocNet(in_channels=3, num=3)

    def forward(self, x, appm, label):
        origin_x = x
        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, _ = self.en4(x)

        x3 = self.multi_dimen_atten3(x3)
        x = self.up3(x4, x3)
        x_up3 = F.interpolate(x, scale_factor=4, mode='trilinear')

        x2 = self.multi_dimen_atten2(x2)
        x = self.up2(x, x2)
        x_up2 = F.interpolate(x, scale_factor=2, mode='trilinear')

        x1 = self.multi_dimen_atten1(x1)
        x = self.up1(x, x1)
        x = torch.cat((x_up3, x_up2, x), dim=1)
        out = self.con_last(x)

        # 在heatmap之前需要crop，这一步没有
        heatmap_list = self.slocnet(out, appm, origin_x)

        result = {
            'msegnet_result': out,
            'heatmap': heatmap_list
        }

        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

