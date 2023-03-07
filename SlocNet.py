import torch
import torch.nn as nn
import torch.nn.functional as F

class FastSmoothSENorm(nn.Module):
    """Se Normalization"""
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
    """采用了Se Normalization的卷积操作"""
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class SlocNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SlocNet, self).__init__()
        self.seconv1 = FastSmoothSeNormConv3d(in_ch, 24, 2, kernel_size=3, stride=1, padding=1)
        self.seconv2 = FastSmoothSeNormConv3d(24, 24, 2, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv3d(24, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, appm, origin_x):
        x = torch.cat((x, appm, origin_x), dim=1)
        out = self.seconv1(x)
        out = self.seconv2(out)
        out = self.conv3(out)

        out = self.sigmoid(out)

        return out


class RoiSlocNet(nn.Module):
    def __init__(self, in_channels, num, depth=72, height=96, width=96):
        super(RoiSlocNet, self).__init__()
        self.slocnets = []
        self.num = num
        for i in range(0, self.num):
            self.slocnets.append(SlocNet(in_channels, 1).cuda())
        self.h_depth = int(depth // 2)
        self.d_height = int(height // 2)
        self.d_width = int(width // 2)

    def forward(self, x, appm, origin_x):
        x = x[:, 2:5, :, :, :]
        location_list = []
        heatmap_list = []
        crop_x, crop_pap, crop_origin_x, location = self.roi_crop(x, appm, origin_x, 0)
        heatmap = self.slocnets[0](crop_x, crop_pap, crop_origin_x)
        location_list.append(location)
        heatmap_list.append(heatmap)

        for i in range(1, self.num):
            crop_x, crop_pap, crop_origin_x, location = self.roi_crop(x, appm, origin_x, 1)
            heatmap = self.slocnets[i][crop_x, crop_pap, crop_origin_x]
            location_list.append(location)
            heatmap_list.append(heatmap)

        return heatmap_list

    def roi_crop(self, x, appm, origin_x, organ_index):
        location = self.center_locate(appm[:, organ_index, :, :, :])
        roi_z, roi_x, roi_y = location

        crop_x = x[:, organ_index, roi_z - self.h_depth: roi_z + self.h_depth,
                       roi_x - self.h_height: roi_x + self.h_height,
                       roi_y - self.h_width: roi_y + self.h_width].detach()

        crop_appm = appm[:, organ_index, roi_z - self.h_depth:roi_z + self.h_depth,
                        roi_x - self.h_height:roi_x + self.h_height,
                        roi_y - self.h_width:roi_y + self.h_width].detach()

        crop_origin_x = origin_x[:, :, roi_z - self.h_depth:roi_z + self.h_depth,
                        roi_x - self.h_height:roi_x + self.h_height,
                        roi_y - self.h_width:roi_y + self.h_width].detach()

        return crop_x, crop_appm, crop_origin_x, location

    def center_locate(self, appm):
        b, d, w, h = appm.shape
        index = torch.argmax(appm)

        b_index = int(index // d // w // h)
        index -= b_index * d * w * h

        z = int(index // w // h)
        index -= z * w * h

        x = int(index // h)
        index -= x * h

        y = int(index)

        x = self.h_height if x < self.h_height else x
        x = 192 - self.h_height if x > 192 - self.h_height else x
        y = self.h_width if y < self.h_width else y
        y = 192 - self.h_width if y > 192 - self.h_width else y
        z = self.h_depth if z < self.h_depth else z
        z = 72 - self.h_depth if z > 72 - self.h_depth else z

        return z, x, y