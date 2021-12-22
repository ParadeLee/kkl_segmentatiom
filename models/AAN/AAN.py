import torch
import torch.nn as nn
import torch.nn.functional as F

class convlayer(nn.Module):
    """3x3 convolution with padding"""
    def __init__(self, in_dims, out_dims, stride, groups=1, dilation=1, is_batchnorm=True):
        if is_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=dilation,
                          groups=groups, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_dims),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=dilation,
                          groups=groups, bias=False, dilation=dilation),
                nn.LeakyReLU(0.2)
            )
    def forward(self, input):
        output = self.conv(input)
        return output

class convup(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, is_batchnorm=True):
        super().__init__()
        self.conv = convlayer(in_size + out_size, out_size, 1, is_batchnorm=is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # outputs2 = F.interpolate(inputs2, size=[inputs1.size(2), inputs1.size(3)], mode='bilinear', align_corners=True)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class ANN(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, is_batchnorm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_batchnorm = is_batchnorm

        filters = [8, 16, 32, 64]

        self.dim_layer = convlayer(in_dim, filters[0], stride=1, is_batchnorm=self.is_batchnorm)
        self.conv1 = convlayer(filters[0], filters[1], stride=2, is_batchnorm=self.is_batchnorm)
        self.conv2 = convlayer(filters[1], filters[2], stride=2, is_batchnorm=self.is_batchnorm)
        self.conv3 = convlayer(filters[2], filters[3], stride=2, is_batchnorm=self.is_batchnorm)

        self.bottom = convlayer(filters[3], filters[3], stride=1, is_batchnorm=self.is_batchnorm)
        self.up1 = convup(filters[3], filters[2], is_batchnorm=self.is_batchnorm)
        self.up2 = convup(filters[2], filters[1], is_batchnorm=self.is_batchnorm)
        self.up3 = convup(filters[1], filters[0], is_batchnorm=self.is_batchnorm)

        self.final = convlayer(filters[0], 1, stride=1, is_batchnorm=self.is_batchnorm)

    def forward(self, x):
        x = self.dim_layer(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv2(conv2)
        up0 = self.bottom(conv3)
        up1 = self.up1(conv3, up0)
        up2 = self.up2(conv2, up1)
        up3 = self.up3(conv1, up2)
        out = self.final(up3)

        return out








