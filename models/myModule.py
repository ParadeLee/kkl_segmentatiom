# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import scipy.misc
from os.path import join as pjoin
from scipy import ndimage
import math
from torchvision import transforms
import cv2
import datetime
from PIL import Image
import time
from torch.nn import init

newsize = 256

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            #x = torch.tensor(x, dtype=torch.float32)
            x = conv(x)

        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class RDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='vanilla'):
        super(RDC, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.decoder = decoder
        self.n_classes = hidden_dim
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2,
                                     self.kernel_size, padding=self.padding, bias=self.bias)

        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim,
                                  self.kernel_size,
                                  padding=self.padding, bias=self.bias)

        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4,
                                      self.kernel_size, padding=self.padding,
                                      bias=self.bias)

        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim,
                                      self.kernel_size,
                                      padding=self.padding, bias=self.bias)

        self.flowconv = nn.Conv2d(self.n_classes * 2, 2,
                                  kernel_size=3, stride=1, padding=1)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _flow_align_module(self, featmap_front, featmap_latter):
        B, C, H, W = featmap_latter.size()
        fuse = torch.cat((featmap_front, self._upsample(featmap_latter, featmap_front)), 1)
        flow = self.flowconv(fuse)
        flow = self._upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        # visualize_flow(flow)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')

        return output

    def forward(self, x_cur, h_pre, c_pre=None):
        if self.decoder == "vanilla":
            h_pre = self._flow_align_module(x_cur, h_pre)
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.vanilla_conv(combined)
            h_cur = torch.relu(combined_conv)

            return h_cur

class UNetDRNN(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):
        super(UNetDRNN, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.kernel_size = kernel_size
        self.bias = bias
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)
        self.RDC2 = RDC(self.n_classes*2, self.kernel_size, bias=self.bias, decoder=self.decoder)

        self.flowconv1 = nn.Conv2d((1024+256)*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d((1024+256)*2, 1024, kernel_size=3, stride=1, padding=1)

        self.flowconv2 = nn.Conv2d(1024+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1)

        self.flowconv3 = nn.Conv2d(256+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)

        self.flowconv4 = nn.Conv2d(128+256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)

        self.flowconv5 = nn.Conv2d(64+128, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)

        ## -------------Score Block in Decoder--------------
        self.score_block1 = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.convp2 = nn.Conv2d(self.n_classes * 5, self.n_classes, kernel_size=3, stride=1, padding=1)
        self.bnp2 = nn.BatchNorm2d(self.n_classes)
        self.relup2 = nn.ReLU(inplace=True)
        self.convp3 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, padding=0)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def _upsample(self, x, y, scale=1):  # the size of x is as y
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size())

    def _flow_align_module(self, featmap_front, featmap_latter, flow):
        B, C, H, W = featmap_latter.size()

        flow = self._upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        # visualize_flow(flow)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')

        return output

    def forward(self, inputs):
        # print(inputs.size())#torch.Size([1, 3, 256, 256])
        ## -------------Transformer-------------
        img = F.upsample(inputs, size=(newsize, newsize), mode='bilinear')
        x = v(img)  # torch.Size([1, 256, 16, 16])
        ## -------------Encoder-------------
        x5 = self.conv1(inputs)  # x1->[1, 64, 256, 256]

        x4 = self.maxpool1(x5)
        x4 = self.conv2(x4)  # x2->[1, 128, 128, 128]

        x3 = self.maxpool2(x4)
        x3 = self.conv3(x3)  # x3->[1, 256, 64, 64]

        x2 = self.maxpool3(x3)
        x2 = self.conv4(x2)  # x4->[1, 512, 32, 32]

        x1 = self.maxpool4(x2)
        x1 = self.conv5(x1)  # x5->[1, 1024, 16, 16]

        [b, c, h, w] = x1.size()
        x = F.upsample(x, size=(h, w), mode='bilinear')

        x1 = torch.cat((x, x1), 1)
        h0 = self._init_cell_state(x1)  # 1/16,512

        if self.decoder == "vanilla":
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            h1 = torch.relu(self.vanilla_conv1(torch.cat([h0_up, x1], dim=1)))
            # print(h1.size())#torch.Size([1, 1024, 11, 11])

            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            h2 = torch.relu(self.vanilla_conv2(torch.cat([h1_up, x2], dim=1)))
            # print(h2.size())#torch.Size([1, 512, 22, 22])

            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            h3 = torch.relu(self.vanilla_conv3(torch.cat([h2_up, x3], dim=1)))
            # print(h3.size())#torch.Size([1, 256, 45, 45])

            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            h4 = torch.relu(self.vanilla_conv4(torch.cat([h3_up, x4], dim=1)))
            # print(h4.size())#torch.Size([1, 128, 90, 90])

            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            h5 = torch.relu(self.vanilla_conv5(torch.cat([h4_up, x5], dim=1)))
            # print(h5.size())#torch.Size([1, 64, 181, 181])
        # sys.exit(0)
        s5 = self.score_block1(h5)  # 1/16,class torch.Size([1, 4, 181, 181])
        s4 = self.score_block2(h4)  # 1/8,class torch.Size([1, 4, 90, 90])
        s3 = self.score_block3(h3)  # 1/4,class torch.Size([1, 4, 45, 45])
        s2 = self.score_block4(h2)  # 1/2,class torch.Size([1, 4, 22, 22])
        s1 = self.score_block5(h1)  # 1,class torch.Size([1, 4, 11, 11])

        if self.decoder == "vanilla":
            p1 = self.RDC(x_cur=s5, h_pre=s1)  # print(p1.size())#torch.Size([1, 4, 181, 181])
            p2 = self.RDC(x_cur=s5, h_pre=s2)  # print(p2.size())#torch.Size([1, 4, 181, 181])
            p3 = self.RDC(x_cur=s5, h_pre=s3)  # print(p3.size())#torch.Size([1, 4, 181, 181])
            p4 = self.RDC(x_cur=s5, h_pre=s4)  # print(p4.size())#torch.Size([1, 4, 181, 181])

        out = torch.cat((p1, p2, p3, p4, s5), 1)

        out = self.convp2(out)
        out = self.relup2(self.bnp2(out))


        return out