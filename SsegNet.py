from torch import nn
from torch import cat
import torch
import cv2
from thop import profile
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair, _triple
import numpy as np

class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            # GAP操作,不是nn.AdaptiveAvgPool
            # 正常的Norm操作也是b,c,-1（h*w或者d*h*w作为一个维度）
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
        x = F.relu(x, inplace=False)
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
        self.relu = nn.ReLU(False)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        #         x = self.dropout(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sn2(x)
        #         x = self.dropout(x)
        x = self.relu(x)
        return x


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv3d(in_channels // 2 + in_channels + 10, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.sn1 = FastSmoothSENorm(out_channels, 2)
        self.sn2 = FastSmoothSENorm(out_channels, 2)
        self.relu = nn.ReLU(False)

    #         self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
    def forward(self, x, x1, edge):
        #         x = self.sample(x)
        x = cat((x, x1, edge), dim=1)
        x = self.conv1(x)
        x = self.sn1(x)
        #         x = self.dropout(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sn2(x)
        #         x = self.dropout(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out += residual
        out = self.relu(out)

        return out


class GatedSpatialConv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(GatedSpatialConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, 'zeros')
        self._gate_conv = nn.Sequential(
            nn.BatchNorm3d(in_channels + 1),
            nn.Conv3d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels + 1, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        # print('input_features shape:', input_features.shape)
        # print('gating_features shape:', gating_features.shape)
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        input_features = (input_features * (alphas + 1))
        return F.conv3d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class SsegNet(nn.Module):
    def __init__(self, ):
        super(SsegNet, self).__init__()
        # origin_x, pap, heatmap
        init_channels = 3
        class_nums = 2
        batch_norm = True
        sample = True

        self.en1 = Downsample_block(init_channels, 48)
        self.en2 = Downsample_block(48, 96)
        self.en3 = Downsample_block(96, 192)
        self.en4 = Downsample_block(192, 384)

        self.res1 = ResBlock(48, 48)
        self.d1 = nn.Conv3d(48, 32, 1)
        self.c2 = nn.Conv3d(96, 1, 1)
        self.gate1 = GatedSpatialConv3d(32, 32)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv3d(32, 16, 1)
        self.c3 = nn.Conv3d(192, 1, 1)
        self.gate2 = GatedSpatialConv3d(16, 16)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv3d(16, 8, 1)
        self.c4 = nn.Conv3d(384, 1, 1)
        self.gate3 = GatedSpatialConv3d(8, 8)
        self.fuse = nn.Conv3d(8, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.cw = nn.Conv3d(2, 1, kernel_size=1, padding=0, bias=False)
        # 卷积的通道数可以修改，暂时不改
        self.expand = nn.Sequential(nn.Conv3d(1, 10, kernel_size=1),
                                    nn.BatchNorm3d(10),
                                    nn.ReLU(inplace=False))

        self.up3 = Upsample_block(384, 192)
        self.up2 = Upsample_block(192, 96)
        self.up1 = Upsample_block(96, 48)
        self.con_last = nn.Conv3d(58, class_nums, 1)

        self.sample = nn.ConvTranspose3d(96, 48, 2, stride=2)

    def forward(self, x, heatmap, appm):
        x_size = x.size()
        x = torch.cat((x, heatmap, appm), dim=1)
        x1 = self.en1(x)
        x2 = self.en2(x1)
        x3 = self.en3(x2)
        x4 = self.en4(x3)

        ss = self.res1(x1)
        ss = self.d1(ss)
        x_2 = self.c2(x2)
        ss1 = self.gate1(ss, x_2)
        ss = self.res2(ss1)
        ss = self.d2(ss)
        x_3 = self.c3(x3)
        ss2 = self.gate2(ss, x_3)
        ss = self.res3(ss2)
        ss = self.d3(ss)
        x_4 = self.c4(x4)
        ss3 = self.gate3(ss, x_4)
        ss = self.fuse(ss3)
        edge_out = self.sigmoid(ss)

        # Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3], x_size[4]))
        for i in range(x_size[0]):
            for j in range(x_size[2]):
                canny[i, 0, j, :, :] = cv2.Canny(im_arr[i, j, :, :], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()


        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)

        x = self.up3(x4, x3, edge)
        x = self.up2(x, x2, edge)
        x = self.up1(x, x1, edge)
        x = torch.cat((x, edge), dim=1)
        out = self.con_last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
