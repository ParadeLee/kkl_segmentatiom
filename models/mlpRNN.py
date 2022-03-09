# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
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
from models.vit import ViT
from .utils.cbam import CBAM
from models.convmlp import ConvMLP
from .utils.cross_attention import NonLocalAttention


np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# newsize = 256
# v = ViT(image_size=newsize, patch_size=16, num_classes=4, dim=16*16,
#         depth=6, heads=16, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1).to(device)

mlp = ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2], channels=64, n_conv_blocks=2)

class MultiFeatureProjection(nn.Module):
    def __init__(self, ):
        super().__init__()


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

class MlpRnn(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):
        super().__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.kernel_size = kernel_size
        self.bias = bias
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        # filters[0]=16 filters[1]=32 filters[2]=64 filters[3]=128 filters[4]=256

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
        #
        # # CTransBlock
        # vis = False
        # img_size = 256
        # patchSize = [32, 16, 8, 4]
        # self.mtc = ChannelTransformer(vis, img_size,
        #                               channel_num=[filters[0], filters[1], filters[2], filters[3]],
        #                               patchSize=patchSize,
        #                               )

        # Spatial - wise Attention
        self.sigmoid = nn.Sigmoid()
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
        )

        ## ------------Unet3+nolocal--------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.CrossAttention4 = NonLocalAttention(channel=filters[3], fusionchannel=self.UpChannels, reduction=2)
        self.CrossAttention3 = NonLocalAttention(channel=filters[2], fusionchannel=self.UpChannels, reduction=2)
        self.CrossAttention2 = NonLocalAttention(channel=filters[1], fusionchannel=self.UpChannels, reduction=2)
        self.CrossAttention1 = NonLocalAttention(channel=filters[0], fusionchannel=self.UpChannels, reduction=2)

        '''skip-4'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=False)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=False)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''skip-3'''
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=False)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''skip-2'''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''skip-1'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------
        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)
        self.RDC2 = RDC(self.n_classes*2, self.kernel_size, bias=self.bias, decoder=self.decoder)

        self.cbam1 = CBAM(filters[4])
        self.cbam2 = CBAM(filters[3])
        self.cbam3 = CBAM(filters[2])
        self.cbam4 = CBAM(filters[1])
        self.cbam5 = CBAM(filters[0])

        # self.flowconv1 = nn.Conv2d((filters[4]+256)*2, 2, kernel_size=3, stride=1, padding=1)
        # self.vanilla_conv1 = nn.Conv2d((filters[4]+256)*2, filters[4], kernel_size=3, stride=1, padding=1)
        self.flowconv1 = nn.Conv2d((filters[4] + 512) * 2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d((filters[4] + 512) * 2, filters[4], kernel_size=3, stride=1, padding=1)

        self.flowconv2 = nn.Conv2d(filters[4]+filters[3], 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(filters[4]+filters[3], filters[3], kernel_size=3, stride=1, padding=1)
        # self.flowconv2 = nn.Conv2d(filters[4]+self.UpChannels, 2, kernel_size=3, stride=1, padding=1)
        # self.vanilla_conv2 = nn.Conv2d(filters[4]+self.UpChannels, filters[3], kernel_size=3, stride=1, padding=1)

        self.flowconv3 = nn.Conv2d(filters[2]+filters[3], 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(filters[2]+filters[3], filters[2], kernel_size=3, stride=1, padding=1)
        # self.flowconv3 = nn.Conv2d(self.UpChannels + filters[3], 2, kernel_size=3, stride=1, padding=1)
        # self.vanilla_conv3 = nn.Conv2d(self.UpChannels + filters[3], filters[2], kernel_size=3, stride=1, padding=1)

        self.flowconv4 = nn.Conv2d(filters[1]+filters[2], 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(filters[1]+filters[2], filters[1], kernel_size=3, stride=1, padding=1)
        # self.flowconv4 = nn.Conv2d(self.UpChannels + filters[2], 2, kernel_size=3, stride=1, padding=1)
        # self.vanilla_conv4 = nn.Conv2d(self.UpChannels + filters[2], filters[1], kernel_size=3, stride=1, padding=1)

        self.flowconv5 = nn.Conv2d(filters[0]+filters[1], 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(filters[0]+filters[1], filters[0], kernel_size=3, stride=1, padding=1)
        # self.flowconv5 = nn.Conv2d(self.UpChannels + filters[1], 2, kernel_size=3, stride=1, padding=1)
        # self.vanilla_conv5 = nn.Conv2d(self.UpChannels + filters[1], filters[0], kernel_size=3, stride=1, padding=1)

        # --------------------------- Depth Refinement Block -------------------------- #
        # DRB 1
        # self.conv_refine1_1 = nn.Conv2d(64, 64, 3, padding=1)
        # self.bn_refine1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine1_1 = nn.PReLU()
        # self.conv_refine1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.bn_refine1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine1_2 = nn.PReLU()
        self.conv_refine1_3 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.bn_refine1_3 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine1_3 = nn.PReLU()
        self.down_2_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.down_2_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # DRB 2
        # self.conv_refine2_1 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn_refine2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine2_1 = nn.PReLU()
        # self.conv_refine2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn_refine2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine2_2 = nn.PReLU()
        self.conv_refine2_3 = nn.Conv2d(filters[1], filters[1], 3, padding=1)
        self.bn_refine2_3 = nn.BatchNorm2d(filters[1], eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine2_3 = nn.PReLU()
        self.conv_r2_1 = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.bn_r2_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu_r2_1 = nn.PReLU()
        # DRB 3
        # self.conv_refine3_1 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn_refine3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine3_1 = nn.PReLU()
        # self.conv_refine3_2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn_refine3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine3_2 = nn.PReLU()
        self.conv_refine3_3 = nn.Conv2d(filters[2], filters[2], 3, padding=1)
        self.bn_refine3_3 = nn.BatchNorm2d(filters[2], eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine3_3 = nn.PReLU()
        self.conv_r3_1 = nn.Conv2d(filters[2], filters[0], 3, padding=1)
        self.bn_r3_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu_r3_1 = nn.PReLU()
        # DRB 4
        # self.conv_refine4_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn_refine4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine4_1 = nn.PReLU()
        # self.conv_refine4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn_refine4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine4_2 = nn.PReLU()
        self.conv_refine4_3 = nn.Conv2d(filters[3], filters[3], 3, padding=1)
        self.bn_refine4_3 = nn.BatchNorm2d(filters[3], eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine4_3 = nn.PReLU()
        self.conv_r4_1 = nn.Conv2d(filters[3], filters[0], 3, padding=1)
        self.bn_r4_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu_r4_1 = nn.PReLU()
        # DRB 5
        # self.conv_refine5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn_refine5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine5_1 = nn.PReLU()
        # self.conv_refine5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn_refine5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        # self.relu_refine5_2 = nn.PReLU()
        self.conv_refine5_3 = nn.Conv2d(filters[4], filters[4], 3, padding=1)
        self.bn_refine5_3 = nn.BatchNorm2d(filters[4], eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine5_3 = nn.PReLU()
        self.conv_r5_1 = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        self.bn_r5_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu_r5_1 = nn.PReLU()

        # -----------------------------  Multi-scale  ----------------------------- #
        # Add new structure: ASPP   Atrous spatial Pyramid Pooling     based on DeepLab v3
        # part0:   1*1*64 Conv
        self.conv5_conv_1 = nn.Conv2d(filters[0], filters[0], 1, padding=0)  # size:  64*64*64
        self.bn5_conv_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu5_conv_1 = nn.ReLU(inplace=True)
        # part1:   3*3*64 Conv
        self.conv5_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)  # size:  64*64*64
        self.bn5_conv = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu5_conv = nn.ReLU(inplace=True)
        # part2:   3*3*64 (dilated=7) Atrous Conv
        self.Atrous_conv_1 = nn.Conv2d(filters[0], filters[0], 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_bn5_1 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_1 = nn.ReLU(inplace=True)
        # part3:   3*3*64 (dilated=5) Atrous Conv
        self.Atrous_conv_2 = nn.Conv2d(filters[0], filters[0], 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_bn5_2 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_2 = nn.ReLU(inplace=True)
        # part4:   3*3*64 (dilated=3) Atrous Conv
        self.Atrous_conv_5 = nn.Conv2d(filters[0], filters[0], 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_bn5_5 = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_5 = nn.ReLU(inplace=True)
        # part5:   Max_pooling                                           # size:  16*16*64
        self.Atrous_pooling = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.Atrous_conv_pool = nn.Conv2d(filters[0], filters[0], 1, padding=0)
        self.Atrous_bn_pool = nn.BatchNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_pool = nn.ReLU(inplace=True)



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
        self.score_block_new = nn.Sequential(
            nn.Conv2d(filters[0]*6, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )


        self.convp2 = nn.Conv2d(self.n_classes * 2, self.n_classes, kernel_size=3, stride=1, padding=1)
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
        return torch.zeros(tensor.size()).to(device)

    def Spatial_attention(self, featmap):
        avg_out = torch.mean(featmap, dim=1, keepdim=True)
        max_out, _ = torch.max(featmap, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spa_add = self.spatial_excitation(cat)
        spa_add = self.sigmoid(spa_add)
        out = torch.mul(featmap, spa_add)
        return out

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
        # img = F.upsample(inputs, size=(newsize, newsize), mode='bilinear')
        # x = v(img)  # torch.Size([1, 256, 16, 16])

        ## --------ConvMLP-----------------------
        x = mlp(inputs)

        ## -------------Encoder-------------
        x5 = self.conv1(inputs)  # x1->[1, 64, 256, 256]

        x4 = self.maxpool1(x5)
        x4 = self.conv2(x4)  # x2->[1, 128, 128, 128]

        x3 = self.maxpool2(x4)
        x3 = self.conv3(x3)  # x3->[1, 256, 64, 64]

        x2 = self.maxpool3(x3)
        x2 = self.conv4(x2)  # x4->[1, 512, 32, 32]

        x1 = self.maxpool4(x2)
        x1_conv = self.conv5(x1)  # x5->[1, 1024, 16, 16]

        # change name
        h1, h2, h3, h4, hd5 = x5, x4, x3, x2, x1_conv
        # skipConnection + crossAttention
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))  # ([1, 64, 22, 22])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))  # ([1, 64, 22, 22])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._upsample(hd5, h4))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._upsample(hd4, h3))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._upsample(hd5, h3))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._upsample(hd3, h2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._upsample(hd4, h2))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._upsample(hd5, h2))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._upsample(hd2, h1))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._upsample(hd3, h1))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._upsample(hd4, h1))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._upsample(hd5, h1))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        x5_f, x4_f, x3_f, x2_f, x1_conv = hd1, hd2, hd3, hd4, hd5
        x2 = self.CrossAttention4(x2, x2_f)
        x3 = self.CrossAttention3(x3, x3_f)
        x4 = self.CrossAttention2(x4, x4_f)
        x5 = self.CrossAttention1(x5, x5_f)

        [_, _, h, w] = x1_conv.size()
        x = F.upsample(x, size=(h, w), mode='bilinear')

        x1 = torch.cat((x, x1_conv), 1)
        h0 = self._init_cell_state(x1)  # 1/16,512

        if self.decoder == "vanilla":
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            h1 = self.cbam1(torch.relu(self.vanilla_conv1(torch.cat([h0_up, x1], dim=1))))
            # print(h1.size())#torch.Size([1, 1024, 11, 11])

            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            h2 = self.cbam2(torch.relu(self.vanilla_conv2(torch.cat([h1_up, x2], dim=1))))
            # print(h2.size())#torch.Size([1, 512, 22, 22])

            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            h3 = self.cbam3(torch.relu(self.vanilla_conv3(torch.cat([h2_up, x3], dim=1))))
            # print(h3.size())#torch.Size([1, 256, 45, 45])

            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            h4 = self.cbam4(torch.relu(self.vanilla_conv4(torch.cat([h3_up, x4], dim=1))))
            # print(h4.size())#torch.Size([1, 128, 90, 90])

            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            h5 = self.cbam5(torch.relu(self.vanilla_conv5(torch.cat([h4_up, x5], dim=1))))
            # print(h5.size())#torch.Size([1, 64, 181, 181])
        # sys.exit(0)
        # s5 = self.score_block1(h5)  # 1/16,class torch.Size([1, 4, 181, 181])
        # s4 = self.score_block2(h4)  # 1/8,class torch.Size([1, 4, 90, 90])
        # s3 = self.score_block3(h3)  # 1/4,class torch.Size([1, 4, 45, 45])
        # s2 = self.score_block4(h2)  # 1/2,class torch.Size([1, 4, 22, 22])
        # s1 = self.score_block5(h1)  # 1,class torch.Size([1, 4, 11, 11])

        # -------- apply DRB --------- #
        # drb 1
        # d1_1 = self.relu_refine1_1(self.bn_refine1_1(self.conv_refine1_1(d1)))
        # d1_2 = self.relu_refine1_2(self.bn_refine1_2(self.conv_refine1_2(d1_1)))
        # d1_2 = d1_2 + h1  # (256x256)*64
        d1_2 = F.interpolate(h5, size=[h3.size(2), h3.size(3)], mode='bilinear', align_corners=True)
        d1_2_0 = d1_2
        d1_3 = self.relu_refine1_3(self.bn_refine1_3(self.conv_refine1_3(d1_2)))
        drb1 = d1_2_0 + d1_3  # (64 x 64)*64

        # drb 2
        # d2_1 = self.relu_refine2_1(self.bn_refine2_1(self.conv_refine2_1(d2)))
        # d2_2 = self.relu_refine2_2(self.bn_refine2_2(self.conv_refine2_2(d2_1)))
        # d2_2 = d2_2 + h2  # (128x128)*128
        d2_2 = F.interpolate(h4, size=[h3.size(2), h3.size(3)], mode='bilinear', align_corners=True)
        d2_2_0 = d2_2
        d2_3 = self.relu_refine2_3(self.bn_refine2_3(self.conv_refine2_3(d2_2)))
        drb2 = d2_2_0 + d2_3
        drb2 = self.relu_r2_1(self.bn_r2_1(self.conv_r2_1(drb2)))  # (64 x 64)*64

        # drb 3
        # d3_1 = self.relu_refine3_1(self.bn_refine3_1(self.conv_refine3_1(d3)))
        # d3_2 = self.relu_refine3_2(self.bn_refine3_2(self.conv_refine3_2(d3_1)))
        # d3_2 = d3_2 + h3  # (64 x 64)*256
        d3_2_0 = h3
        d3_3 = self.relu_refine3_3(self.bn_refine3_3(self.conv_refine3_3(h3)))
        drb3 = d3_2_0 + d3_3
        drb3 = self.relu_r3_1(self.bn_r3_1(self.conv_r3_1(drb3)))  # (64 x 64)*64

        # drb 4
        # d4_1 = self.relu_refine4_1(self.bn_refine4_1(self.conv_refine4_1(d4)))
        # d4_2 = self.relu_refine4_2(self.bn_refine4_2(self.conv_refine4_2(d4_1)))
        # d4_2 = d4_2 + h4  # (32 x 32)*512
        d4_2 = F.interpolate(h2, size=[h3.size(2), h3.size(3)], mode='bilinear', align_corners=True)
        d4_2_0 = d4_2
        d4_3 = self.relu_refine4_3(self.bn_refine4_3(self.conv_refine4_3(d4_2)))
        drb4 = d4_2_0 + d4_3
        drb4 = self.relu_r4_1(self.bn_r4_1(self.conv_r4_1(drb4)))  # (64 x 64)*64

        # drb 5
        # d5_1 = self.relu_refine5_1(self.bn_refine5_1(self.conv_refine5_1(d5)))
        # d5_2 = self.relu_refine5_2(self.bn_refine5_2(self.conv_refine5_2(d5_1)))
        # d5_2 = d5_2 + h5  # (16 x 16)*64
        d5_2 = F.interpolate(h1, size=[h3.size(2), h3.size(3)], mode='bilinear', align_corners=True)
        d5_2_0 = d5_2
        d5_3 = self.relu_refine5_3(self.bn_refine5_3(self.conv_refine5_3(d5_2)))
        drb5 = d5_2_0 + d5_3
        drb5 = self.relu_r5_1(self.bn_r5_1(self.conv_r5_1(drb5)))  # (64 x 64)*64

        drb_fusion = drb1 + drb2 + drb3 + drb4 + drb5

        # --------------------- obtain multi-scale ----------------------- #
        f1 = self.relu5_conv_1(self.bn5_conv_1(self.conv5_conv_1(drb_fusion)))
        f2 = self.relu5_conv(self.bn5_conv(self.conv5_conv(drb_fusion)))
        f3 = self.Atrous_relu_1(self.Atrous_bn5_1(self.Atrous_conv_1(drb_fusion)))
        f4 = self.Atrous_relu_2(self.Atrous_bn5_2(self.Atrous_conv_2(drb_fusion)))
        f5 = self.Atrous_relu_5(self.Atrous_bn5_5(self.Atrous_conv_5(drb_fusion)))
        f6 = F.upsample(
            self.Atrous_relu_pool(
                self.Atrous_bn_pool(self.Atrous_conv_pool(self.Atrous_pooling(self.Atrous_pooling(drb_fusion))))),
            size=[f1.size(2), f1.size(3)], mode='bilinear')

        fusion = torch.cat([f1, f2, f3, f4, f5, f6], dim=0)  # 6x64x64x64

        input = torch.cat(torch.chunk(fusion, 6, dim=0), dim=1)  # 1x64x64x64

        s5 = self.score_block1(h5)  # 1/16,class torch.Size([1, 4, 181, 181])
        input = self.score_block_new(input)  # 1x4x64x64
        s5 = self.Spatial_attention(s5)
        input = self.Spatial_attention(input)

        p = self.RDC(x_cur=s5, h_pre=input)

        out = torch.cat((p, s5), 1)

        out = self.convp2(out)
        out = self.relup2(self.bnp2(out))


        return out