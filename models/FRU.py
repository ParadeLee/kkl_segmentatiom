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
import cv2
import datetime
from PIL import Image
np.set_printoptions(threshold=np.inf)
#f = open('log.txt','w')
__all__ = ['UNetDRNN', 'VGG16DRNN', 'ResNet50DRNN', "UNet3", "UNetD"]
    
class RDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='GRU'):
        super(RDC, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.decoder = decoder
        self.n_classes = hidden_dim
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size, padding=self.padding, bias=self.bias)
        
        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                  padding=self.padding, bias=self.bias)
                                  
        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size, padding=self.padding, bias=self.bias)
        
        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                      padding=self.padding, bias=self.bias)
        self.flowconv = nn.Conv2d(self.n_classes*2, 2, kernel_size=3, stride=1, padding=1)
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
        #visualize_flow(flow)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:,:,:,0] / max(W-1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,:,:,1] / max(H-1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')
        #######################
        '''fuse = torch.cat((featmap_latter, self._upsample(featmap_latter, output)), 1)
        flow = self.flowconv(fuse)
        flow = self._upsample(flow, output)
        flow = flow.permute(0, 2, 3, 1)
        if(H>100):
            visualize_flow(flow)'''
        #visualize_flow(vgrid_scaled)
        
        return output

    def forward(self, x_cur, h_pre, c_pre=None):
        if self.decoder == "LSTM":
            c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            FAM_flow = self._flow_align_module(x_cur, h_pre)
            h_pre_up = F.interpolate(FAM_flow, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.lstm_catconv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_cur = f * c_pre_up + i * g
            h_cur = o * torch.tanh(c_cur)

            return h_cur, c_cur

        elif self.decoder == "GRU":
            h_pre_flow = self._flow_align_module(x_cur, h_pre)
            h_pre_up = F.interpolate(h_pre_flow, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.gru_catconv(combined)
            
            cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv(torch.cat([x_cur, r * h_pre_up], dim=1)))
            h_cur = z * h_pre_up + (1 - z) * h_hat

            return h_cur

        elif self.decoder == "vanilla":
            h_pre = self._flow_align_module(x_cur, h_pre)
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.vanilla_conv(combined)
            h_cur = torch.relu(combined_conv)

            return h_cur

class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=4, auxseg=False):
        super(SelFuseFeature, self).__init__()
        
        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)
        

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0
        
        #print(df.shape) #torch.Size([1, 2, 217, 181])
        #visualize_flow(df)
        
        scale = 1.
        
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)

        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1)#.transpose(1, 2)
        grid_ = grid + 0.
        grid[...,0] = 2*grid_[..., 0] / (H-1) - 1
        grid[...,1] = 2*grid_[..., 1] / (W-1) - 1

        # features = []
        select_x = x.clone()
        select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')
        
        return select_x, df
        

"""
Implementation code for CRDN with U-Net D3+-backbone (UNetD3+RNN).
"""

class UNetDRNN(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, decoder="vanilla",
                 bias=True, is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):
        super(UNetDRNN, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.kernel_size= kernel_size
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
        self.flowconv1 = nn.Conv2d(1024*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d(1024*2, 1024, kernel_size=3, stride=1, padding=1)
        
        self.flowconv2 = nn.Conv2d(1024+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1)
        
        self.flowconv3 = nn.Conv2d(256+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        
        self.flowconv4 = nn.Conv2d(128+256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)
        
        self.flowconv5 = nn.Conv2d(64+128, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.gru_catconv1 = nn.Conv2d(1024*2, 1024*2, kernel_size=3, stride=1, padding=1)
        self.gru_catconv2 = nn.Conv2d(1024+512, 1024+512, kernel_size=3, stride=1, padding=1)
        self.gru_catconv3 = nn.Conv2d(256+512, 256+512, kernel_size=3, stride=1, padding=1)
        self.gru_catconv4 = nn.Conv2d(128+256, 128+256, kernel_size=3, stride=1, padding=1)
        self.gru_catconv5 = nn.Conv2d(64+128, 64+128, kernel_size=3, stride=1, padding=1)
        
        self.gru_conv1 = nn.Conv2d(1024*2, 1024, kernel_size=3, stride=1, padding=1)
        self.gru_conv2 = nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1)
        self.gru_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        self.gru_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)
        self.gru_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.reduceh1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.reduceh2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reduceh3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.reduceh4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        
        self.lstm_catconv1 = nn.Conv2d(1024*2, 1024*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv2 = nn.Conv2d(1024+512, 512*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv3 = nn.Conv2d(512+256, 256*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv4 = nn.Conv2d(256+128, 128*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv5 = nn.Conv2d(128+64, 64*4, kernel_size=3, stride=1, padding=1)
        
        self.reducec1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.reducec2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reducec3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.reducec4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
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

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
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

        '''stage 2d '''
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

        '''stage 1d'''
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

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        
        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)
        self.flowconv1 = nn.Conv2d(1024*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d(1024*2, 1024, kernel_size=3, stride=1, padding=1)
        
        self.flowconv2 = nn.Conv2d(1024+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1)
        
        self.flowconv3 = nn.Conv2d(256+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        
        self.flowconv4 = nn.Conv2d(128+256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)
        
        self.flowconv5 = nn.Conv2d(64+128, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.score_block1 = nn.Sequential(

            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )
        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)
        
        self.convp2 = nn.Conv2d(self.n_classes * 5, filters[4], kernel_size=3, stride=1, padding=1)
        self.bnp2 = nn.BatchNorm2d(filters[4])
        self.relup2 = nn.ReLU(inplace=True)
        self.convp3 = nn.Conv2d(filters[4], self.n_classes, kernel_size=1, stride=1, padding=0)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                
        # Direct Field
        self.fuse_conv = nn.Sequential(nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(self.n_classes),
                                    nn.ReLU(inplace=True),
                                    )
        self.ConvDf_1x1 = nn.Conv2d(self.n_classes, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = SelFuseFeature(self.n_classes, auxseg=auxseg, shift_n=shift_n)

        self.Conv_1x1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, padding=0)
      
    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size())
        
    def _flow_align_module(self, featmap_front, featmap_latter, flow):
        B, C, H, W = featmap_latter.size()
        
        flow = self._upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        #visualize_flow(flow)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:,:,:,0] / max(W-1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,:,:,1] / max(H-1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')

        return output
        
    def _upsample(self, x, y, scale=1): # the size of x is as y
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')    
        
    def get_brainweb_colormap(self):
        return np.asarray([[0, 0, 0], [255, 255, 255], [92, 179, 179], [221, 218, 93]])
    
    def decode_segmap(self, label_mask):
        label_colors = self.get_brainweb_colormap()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colors[ll, 0]
            g[label_mask == ll] = label_colors[ll, 1]
            b[label_mask == ll] = label_colors[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb.astype(np.uint8)
        
    def draw(self, input_data, img_name):
        seg_out = F.softmax(input_data, dim=1)
        _, pred = torch.max(seg_out, 1)
        pred = pred.squeeze(0).numpy()
        '''print(pred)
        print(pred.shape)'''
        #pred = np.squeeze(input_data.max(1)[1].cpu().numpy())
        decoded = self.decode_segmap(pred)
        scipy.misc.imsave(pjoin('./training_vis','{}.bmp'.format(img_name)),decoded)    
    
    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->[1, 64, 181, 181]

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->[1, 128, 90, 90]

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->[1, 256, 45, 45]

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->[1, 512, 22, 22]

        h5 = self.maxpool4(h4)
        h5 = self.conv5(h5)  # h5->[1, 1024, 11, 11]
        hd5 = h5
        
        x1 = h5# 1/16,class torch.Size([1, 1024, 11, 11])
        x2 = h4# 1/8,class torch.Size([1, 512, 22, 22])
        x3 = h3# 1/4,class torch.Size([1, 256, 45, 45])
        x4 = h2# 1/2,class torch.Size([1, 128, 90, 90])
        x5 = h1# 1,class torch.Size([1, 64, 181, 181])
        
        h0 = self._init_cell_state(x1)
        
        if self.decoder == "vanilla":
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            h1 = torch.relu(self.vanilla_conv1(torch.cat([h0_up, x1], dim=1)))#torch.Size([1, 512, 11, 11])
            #print(h1.size())#torch.Size([1, 1024, 11, 11])
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            h2 = torch.relu(self.vanilla_conv2(torch.cat([h1_up, x2], dim=1)))
            #print(h2.size())#torch.Size([1, 512, 22, 22])
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            h3 = torch.relu(self.vanilla_conv3(torch.cat([h2_up, x3], dim=1)))
            #print(h3.size())#torch.Size([1, 256, 45, 45])
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            h4 = torch.relu(self.vanilla_conv4(torch.cat([h3_up, x4], dim=1)))
            #print(h4.size())#torch.Size([1, 128, 90, 90])
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            h5 = torch.relu(self.vanilla_conv5(torch.cat([h4_up, x5], dim=1)))
            #print(h5.size())#torch.Size([1, 64, 181, 181])   
        
        if self.decoder == "LSTM":
            c0 = self._init_cell_state(h0)
            
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 1024, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c_pre_up = F.interpolate(c0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            c1 = f * c_pre_up + i * g
            h1 = o * torch.tanh(c1)
            
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 512, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c1_up = F.interpolate(c1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec1(c1_up)
            c2 = f * c_re + i * g
            h2 = o * torch.tanh(c2)
            
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 256, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c2_up = F.interpolate(c2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec2(c2_up)
            c3 = f * c_re + i * g
            h3 = o * torch.tanh(c3)
            
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 128, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c3_up = F.interpolate(c3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec3(c3_up)
            c4 = f * c_re + i * g
            h4 = o * torch.tanh(c4)
            
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 64, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c4_up = F.interpolate(c4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec4(c4_up)
            c5 = f * c_re + i * g
            h5 = o * torch.tanh(c5)    
            
        elif self.decoder == "GRU":
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_r, cc_z = torch.split(combined_conv, 1024, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv1(torch.cat([x1, r * h0_up], dim=1)))
            h1 = z * h0_up + (1 - z) * h_hat
            #h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [1024,512], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv2(torch.cat([x2, r * h1_up], dim=1)))
            h1_re = self.reduceh1(h1_up)
            h2 = z * h1_re + (1 - z) * h_hat
            #h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [512,256], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv3(torch.cat([x3, r * h2_up], dim=1)))
            h2_re = self.reduceh2(h2_up)
            h3 = z * h2_re + (1 - z) * h_hat
            #h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [256,128], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv4(torch.cat([x4, r * h3_up], dim=1)))
            h3_re = self.reduceh3(h3_up)
            h4 = z * h3_re + (1 - z) * h_hat
            #h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [128,64], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv5(torch.cat([x5, r * h4_up], dim=1)))
            h4_re = self.reduceh4(h4_up)
            h5 = z * h4_re + (1 - z) * h_hat    
        
        #sys.exit(0)
        x1, x2, x3, x4, x5 = h5, h4, h3, h2, h1
        h1, h2, h3, h4, hd5 = x1, x2, x3, x4, x5
        
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))#([1, 64, 22, 22])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))#([1, 64, 22, 22])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._upsample(hd5, h4))))
        #print(h1_PT_hd4.size(), h2_PT_hd4.size(), h3_PT_hd4.size(), h4_Cat_hd4.size(), hd5_UT_hd4.size())
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels
            
        #print(hd4.size())#torch.Size([1, 320, 22, 22])
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._upsample(hd4, h3))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._upsample(hd5, h3))))
        #print(h1_PT_hd3.size(), h2_PT_hd3.size(), h3_Cat_hd3.size(), hd4_UT_hd3.size(), hd5_UT_hd3.size())
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels
        #print(hd3.size())#torch.Size([1, 320, 45, 45])
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._upsample(hd3, h2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._upsample(hd4, h2))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._upsample(hd5, h2))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels
        #print(hd2.size())#torch.Size([1, 320, 90, 90])
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._upsample(hd2, h1))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._upsample(hd3, h1))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._upsample(hd4, h1))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._upsample(hd5, h1))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        finalseg = hd1
        h1 = self.score_block1(hd1)# 1/16,class torch.Size([1, 4, 181, 181])
        h2 = self.score_block2(hd2)# 1/8,class torch.Size([1, 4, 90, 90])
        h3 = self.score_block3(hd3)# 1/4,class torch.Size([1, 4, 45, 45])
        h4 = self.score_block4(hd4)# 1/2,class torch.Size([1, 4, 22, 22])
        h5 = self.score_block5(hd5)# 1,class torch.Size([1, 4, 11, 11])
        
        '''self.draw(h1, 'h1')
        self.draw(h2, 'h2')
        self.draw(h3, 'h3')
        self.draw(h4, 'h4')
        self.draw(h5, 'h5')'''
        
        if self.decoder == "vanilla" or self.decoder == "GRU":        
            p1 = self.RDC(x_cur=h1, h_pre=h5);#print(p1.size())#torch.Size([1, 4, 181, 181])
            p2 = self.RDC(x_cur=h1, h_pre=h4);#print(p2.size())#torch.Size([1, 4, 181, 181])
            p3 = self.RDC(x_cur=h1, h_pre=h3);#print(p3.size())#torch.Size([1, 4, 181, 181])
            p4 = self.RDC(x_cur=h1, h_pre=h2);#print(p4.size())#torch.Size([1, 4, 181, 181])
        elif self.decoder == "LSTM": 
            c0 = self._init_cell_state(h5)
            p1, c1 = self.RDC(x_cur=h1, h_pre=h5, c_pre=c0);#print(p1.size())
            p2, c2 = self.RDC(x_cur=h1, h_pre=h4, c_pre=c1);#print(p1.size())
            p3, c3 = self.RDC(x_cur=h1, h_pre=h3, c_pre=c2);#print(p1.size())
            p4, c4 = self.RDC(x_cur=h1, h_pre=h2, c_pre=c3);#print(p1.size())
        #sys.exit(0)
        out = torch.cat((h1, p1, p2, p3, p4), 1);#print(out.size())#torch.Size([1, 20, 181, 181])
        '''self.draw(p1, 's1')
        self.draw(p2, 's2')
        self.draw(p3, 's3')
        self.draw(p4, 's4')'''
        
        out = self.convp2(out);#print(out.size())
        out = self.relup2(self.bnp2(out));#print(out.size())
        out = self.convp3(out);#print(out.size())
        out = self._upsample(out, h1);#print(out.size())#torch.Size([1, 4, 181, 181])
        initseg = out
        #self.draw(out, 'initseg')    
        
        # Direct Field
        #out = finalseg
        shift_n = 5
        for _ in range(shift_n):
            df = self.ConvDf_1x1(out)#torch.Size([1, 2, 181, 217])
            out, augdf = self.SelDF(out, df)#torch.Size([1, 64, 181, 217]), torch.Size([1, 2, 181, 217])

        select_x = self.fuse_conv(torch.cat([initseg, out], dim=1))
        out = select_x # torch.Size([1, 4, 181, 217])
        d1 = self.Conv_1x1(out)#torch.Size([1, 4, 181, 217])
        
        #self.draw(d1, 'finalseg')  
        #self.draw(out, 'out')  
        #self.draw(auxseg, 'auxseg')  
        
        #sys.exit(0)
        return [d1, df, initseg]

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear',
                                 align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))

"""
Implementation code for CRDN with VGG16 (VGG26RNN).
"""


class VGG16DRNN(nn.Module):
    def __init__(self, input_channel, n_classes, kernel_size, decoder="LSTM", bias=True, is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):

        super(VGG16DRNN, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.decoder = decoder
        self.bias = bias

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_block4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_block5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        filters = [64, 128, 256, 512, 512]
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        self.score_block1 = nn.Sequential(

            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(self.UpChannels, n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )
        
        '''stage 4d'''
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

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
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

        '''stage 2d '''
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

        '''stage 1d'''
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

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)
        self.flowconv1 = nn.Conv2d(512*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        
        self.flowconv2 = nn.Conv2d(512*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        
        self.flowconv3 = nn.Conv2d(256+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        
        self.flowconv4 = nn.Conv2d(128+256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)
        
        self.flowconv5 = nn.Conv2d(64+128, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.gru_catconv1 = nn.Conv2d(512*2, 512*2, kernel_size=3, stride=1, padding=1)
        self.gru_catconv2 = nn.Conv2d(512*2, 512*2, kernel_size=3, stride=1, padding=1)
        self.gru_catconv3 = nn.Conv2d(256+512, 256+512, kernel_size=3, stride=1, padding=1)
        self.gru_catconv4 = nn.Conv2d(128+256, 128+256, kernel_size=3, stride=1, padding=1)
        self.gru_catconv5 = nn.Conv2d(64+128, 64+128, kernel_size=3, stride=1, padding=1)
        
        
        self.gru_conv1 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        self.gru_conv2 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        self.gru_conv3 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        self.gru_conv4 = nn.Conv2d(128+256, 128, kernel_size=3, stride=1, padding=1)
        self.gru_conv5 = nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.reduceh1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.reduceh2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reduceh3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.reduceh4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        
        self.lstm_catconv1 = nn.Conv2d(512*2, 512*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv2 = nn.Conv2d(512*2, 512*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv3 = nn.Conv2d(256+512, 256*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv4 = nn.Conv2d(128+256, 128*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv5 = nn.Conv2d(64+128, 64*4, kernel_size=3, stride=1, padding=1)
        
        self.reducec1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.reducec2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reducec3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.reducec4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        
        self.convp2 = nn.Conv2d(self.n_classes * 5, 256, kernel_size=3, stride=1, padding=1)
        self.bnp2 = nn.BatchNorm2d(256)
        self.relup2 = nn.ReLU(inplace=True)
        self.convp3 = nn.Conv2d(256, self.n_classes, kernel_size=1, stride=1, padding=0)
        
        # Direct Field
        self.fuse_conv = nn.Sequential(nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(self.n_classes),
                                    nn.ReLU(inplace=True),
                                    )
        self.ConvDf_1x1 = nn.Conv2d(self.n_classes, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = SelFuseFeature(self.n_classes, auxseg=auxseg, shift_n=shift_n)

        self.Conv_1x1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, padding=0)

        
     
    def _flow_align_module(self, featmap_front, featmap_latter, flow):
        B, C, H, W = featmap_latter.size()
        flow = self._upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:,:,:,0] / max(W-1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,:,:,1] / max(H-1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')
        return output
     
    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        
    def forward(self, input, cell_state=None):

        # Encode
        conv1 = self.conv_block1(input)  # 1,64
        conv2 = self.conv_block2(conv1)  # 1/2,128
        conv3 = self.conv_block3(conv2)  # 1/4,256
        conv4 = self.conv_block4(conv3)  # 1/8,512
        conv5 = self.conv_block5(conv4)  # 1/16,512
        
        x1 = conv5# 1/16,class torch.Size([1, 4, 11, 13])
        x2 = conv4# 1/8,class torch.Size([1, 4, 22, 27])
        x3 = conv3# 1/4,class torch.Size([1, 4, 45, 54])
        x4 = conv2# 1/2,class torch.Size([1, 4, 90, 108])
        x5 = conv1# 1,class torch.Size([1, 4, 181, 217])

        h0 = self._init_cell_state(x1)  # 1/16,512

        # Decode
        if self.decoder == "LSTM":
            # init c0
            if cell_state is not None:
                raise NotImplementedError()
            else:
                c0 = self._init_cell_state(h0)

            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 512, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c_pre_up = F.interpolate(c0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            c1 = f * c_pre_up + i * g
            h1 = o * torch.tanh(c1)
            
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 512, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c1_up = F.interpolate(c1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec1(c1_up)
            c2 = f * c_re + i * g
            h2 = o * torch.tanh(c2)
            
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 256, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c2_up = F.interpolate(c2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec2(c2_up)
            c3 = f * c_re + i * g
            h3 = o * torch.tanh(c3)
            
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 128, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c3_up = F.interpolate(c3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec3(c3_up)
            c4 = f * c_re + i * g
            h4 = o * torch.tanh(c4)
            
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 64, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c4_up = F.interpolate(c4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec4(c4_up)
            c5 = f * c_re + i * g
            h5 = o * torch.tanh(c5)
            
            
        elif self.decoder == "GRU":
            
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_r, cc_z = torch.split(combined_conv, 512, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv1(torch.cat([x1, r * h0_up], dim=1)))
            h1 = z * h0_up + (1 - z) * h_hat
            #h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_r, cc_z = torch.split(combined_conv, 512, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv2(torch.cat([x2, r * h1_up], dim=1)))
            h1_re = self.reduceh1(h1_up)
            h2 = z * h1_re + (1 - z) * h_hat
            #h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [512,256], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv3(torch.cat([x3, r * h2_up], dim=1)))
            h2_re = self.reduceh2(h2_up)
            h3 = z * h2_re + (1 - z) * h_hat
            #h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [256,128], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv4(torch.cat([x4, r * h3_up], dim=1)))
            h3_re = self.reduceh3(h3_up)
            h4 = z * h3_re + (1 - z) * h_hat
            #h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [128,64], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv5(torch.cat([x5, r * h4_up], dim=1)))
            h4_re = self.reduceh4(h4_up)
            h5 = z * h4_re + (1 - z) * h_hat
            #h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class


        elif self.decoder == "vanilla":
            #print(h0.size(),x1.size())
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            h1 = torch.relu(self.vanilla_conv1(torch.cat([h0_up, x1], dim=1)))
            #h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            #print(h1.size(),x2.size())
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            h2 = torch.relu(self.vanilla_conv2(torch.cat([h1_up, x2], dim=1)))
            #h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            #print(h2.size(),x3.size())
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            h3 = torch.relu(self.vanilla_conv3(torch.cat([h2_up, x3], dim=1)))
            #h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            h4 = torch.relu(self.vanilla_conv4(torch.cat([h3_up, x4], dim=1)))
            #h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            h5 = torch.relu(self.vanilla_conv5(torch.cat([h4_up, x5], dim=1)))
            #h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class
            # Direct Field
            
        x1, x2, x3, x4, x5 = h5, h4, h3, h2, h1
        h1, h2, h3, h4, hd5 = x1, x2, x3, x4, x5
        
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))#([1, 64, 22, 22])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))#([1, 64, 22, 22])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._upsample(hd5, h4))))
        #print(h1_PT_hd4.size(), h2_PT_hd4.size(), h3_PT_hd4.size(), h4_Cat_hd4.size(), hd5_UT_hd4.size())
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels
            
        #print(hd4.size())#torch.Size([1, 320, 22, 22])
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._upsample(hd4, h3))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._upsample(hd5, h3))))
        #print(h1_PT_hd3.size(), h2_PT_hd3.size(), h3_Cat_hd3.size(), hd4_UT_hd3.size(), hd5_UT_hd3.size())
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels
        #print(hd3.size())#torch.Size([1, 320, 45, 45])
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._upsample(hd3, h2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._upsample(hd4, h2))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._upsample(hd5, h2))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels
        #print(hd2.size())#torch.Size([1, 320, 90, 90])
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._upsample(hd2, h1))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._upsample(hd3, h1))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._upsample(hd4, h1))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._upsample(hd5, h1))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        #finalseg = hd1
        h1 = self.score_block1(hd1)# 1/16,class torch.Size([1, 4, 181, 181])
        h2 = self.score_block2(hd2)# 1/8,class torch.Size([1, 4, 90, 90])
        h3 = self.score_block3(hd3)# 1/4,class torch.Size([1, 4, 45, 45])
        h4 = self.score_block4(hd4)# 1/2,class torch.Size([1, 4, 22, 22])
        h5 = self.score_block5(hd5)# 1,class torch.Size([1, 4, 11, 11])
        
        if self.decoder == "vanilla" or self.decoder == "GRU":        
            p1 = self.RDC(x_cur=h1, h_pre=h5);#print(p1.size())#torch.Size([1, 4, 181, 181])
            p2 = self.RDC(x_cur=h1, h_pre=h4);#print(p2.size())#torch.Size([1, 4, 181, 181])
            p3 = self.RDC(x_cur=h1, h_pre=h3);#print(p3.size())#torch.Size([1, 4, 181, 181])
            p4 = self.RDC(x_cur=h1, h_pre=h2);#print(p4.size())#torch.Size([1, 4, 181, 181])
        elif self.decoder == "LSTM": 
            c0 = self._init_cell_state(h5)
            p1, c1 = self.RDC(x_cur=h1, h_pre=h5, c_pre=c0);#print(p1.size())
            p2, c2 = self.RDC(x_cur=h1, h_pre=h4, c_pre=c1);#print(p1.size())
            p3, c3 = self.RDC(x_cur=h1, h_pre=h3, c_pre=c2);#print(p1.size())
            p4, c4 = self.RDC(x_cur=h1, h_pre=h2, c_pre=c3);#print(p1.size())
        
        out = torch.cat((h1, p1, p2, p3, p4), 1);#print(out.size())#torch.Size([1, 20, 181, 181])
        
        out = self.convp2(out);#print(out.size())
        out = self.relup2(self.bnp2(out));#print(out.size())
        out = self.convp3(out);#print(out.size())
        out = self._upsample(out, h1);#print(out.size())#torch.Size([1, 4, 181, 181])
        initseg = out
        # Direct Field
        #out = finalseg
        shift_n = 5
        for _ in range(shift_n):
            df = self.ConvDf_1x1(out)#torch.Size([1, 2, 181, 217])
            out, augdf = self.SelDF(out, df)#torch.Size([1, 64, 181, 217]), torch.Size([1, 2, 181, 217])

        select_x = self.fuse_conv(torch.cat([initseg, out], dim=1))
        out = select_x # torch.Size([1, 4, 181, 217])
        d1 = self.Conv_1x1(out)#torch.Size([1, 4, 181, 217])

        return [d1, df, initseg]

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size())

"""
Implementation code for CRDN with ResNet (ResNetRNN).
"""

class ResNetRNN(nn.Module):

    def __init__(self, block, layers, input_channel=1, n_classes=4, kernel_size=3, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):
   
        super(ResNetRNN, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.bias = bias
        self.kernel_size = kernel_size
        self.inplanes = 64
        self.decoder = decoder
        
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        
        filters = [64, 256, 512, 1024, 2048]
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        self.conv1_score_block = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.conv2_score_block = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.conv3_score_block = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.conv4_score_block = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.conv5_score_block = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 3, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )
        
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
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

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
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

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
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

        '''stage 1d'''
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

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)

        self.flowconv1 = nn.Conv2d(2048*2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d(2048*2, 2048, kernel_size=3, stride=1, padding=1)
        
        self.flowconv2 = nn.Conv2d(1024+2048, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(1024+2048, 1024, kernel_size=3, stride=1, padding=1)
        
        self.flowconv3 = nn.Conv2d(512+1024, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(512+1024, 512, kernel_size=3, stride=1, padding=1)
        
        self.flowconv4 = nn.Conv2d(256+512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        
        self.flowconv5 = nn.Conv2d(64+256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64+256, 64, kernel_size=3, stride=1, padding=1)
        
        self.gru_catconv1 = nn.Conv2d(2048*2, 2048*2, kernel_size=3, stride=1, padding=1)
        self.gru_catconv2 = nn.Conv2d(1024+2048, 1024+2048, kernel_size=3, stride=1, padding=1)
        self.gru_catconv3 = nn.Conv2d(512+1024, 512+1024, kernel_size=3, stride=1, padding=1)
        self.gru_catconv4 = nn.Conv2d(256+512, 256+512, kernel_size=3, stride=1, padding=1)
        self.gru_catconv5 = nn.Conv2d(64+256, 64+256, kernel_size=3, stride=1, padding=1)
        
        self.gru_conv1 = nn.Conv2d(2048*2, 2048, kernel_size=3, stride=1, padding=1)
        self.gru_conv2 = nn.Conv2d(1024+2048, 1024, kernel_size=3, stride=1, padding=1)
        self.gru_conv3 = nn.Conv2d(512+1024, 512, kernel_size=3, stride=1, padding=1)
        self.gru_conv4 = nn.Conv2d(256+512, 256, kernel_size=3, stride=1, padding=1)
        self.gru_conv5 = nn.Conv2d(64+256, 64, kernel_size=3, stride=1, padding=1)
        
        self.reduceh1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.reduceh2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.reduceh3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reduceh4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        
        self.lstm_catconv1 = nn.Conv2d(2048*2, 2048*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv2 = nn.Conv2d(1024+2048, 1024*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv3 = nn.Conv2d(512+1024, 512*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv4 = nn.Conv2d(256+512, 256*4, kernel_size=3, stride=1, padding=1)
        self.lstm_catconv5 = nn.Conv2d(64+256, 64*4, kernel_size=3, stride=1, padding=1)
        
        self.reducec1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.reducec2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.reducec3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reducec4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        
        self.convp2 = nn.Conv2d(self.n_classes * 5, 256, kernel_size=3, stride=1, padding=1)
        self.bnp2 = nn.BatchNorm2d(256)
        self.relup2 = nn.ReLU(inplace=True)
        self.convp3 = nn.Conv2d(256, self.n_classes, kernel_size=1, stride=1, padding=0)
        
        # Direct Field
        self.ConvDf_1x1 = nn.Conv2d(self.n_classes, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = SelFuseFeature(self.n_classes, shift_n=shift_n, n_class=4, auxseg=auxseg)

        self.Conv_1x1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, padding=0)

        self.fuse_conv = nn.Sequential(nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(self.n_classes),
                                    nn.ReLU(inplace=True),
                                    )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _flow_align_module(self, featmap_front, featmap_latter, flow):
        B, C, H, W = featmap_latter.size()
        flow = self._upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:,:,:,0] / max(W-1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,:,:,1] / max(H-1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')
        return output
        
    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
       
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input, cell_state=None):
        x = self.conv1(input)
        x = self.bn1(x)
        down1 = self.relu(x)  # 1, 64
        down2 = self.maxpool(down1)

        down2 = self.layer1(down2)  # 1/2, 256
        down3 = self.layer2(down2)  # 1/4, 512
        down4 = self.layer3(down3)  # 1/8, 1024
        down5 = self.layer4(down4)  # 1/16, 2048

        x1 = down5  # 1/16, class torch.Size([1, 2048, 12, 14])
        x2 = down4  # 1/8, class  torch.Size([1, 1024, 23, 28])
        x3 = down3  # 1/4, class  torch.Size([1, 512, 46, 55])
        x4 = down2  # 1/2, class  torch.Size([1, 256, 91, 109])
        x5 = down1  # 1, class    torch.Size([1, 64, 181, 217])

        h0 = self._init_cell_state(x1)  # 1/16, class

        # Decode
        if self.decoder == "LSTM":
            # init c0
            if cell_state is not None:
                raise NotImplementedError()
            else:
                c0 = self._init_cell_state(h0)

            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 2048, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c_pre_up = F.interpolate(c0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            c1 = f * c_pre_up + i * g
            h1 = o * torch.tanh(c1)
            
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 1024, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c1_up = F.interpolate(c1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec1(c1_up)
            c2 = f * c_re + i * g
            h2 = o * torch.tanh(c2)
            
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 512, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c2_up = F.interpolate(c2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec2(c2_up)
            c3 = f * c_re + i * g
            h3 = o * torch.tanh(c3)
            
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 256, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c3_up = F.interpolate(c3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec3(c3_up)
            c4 = f * c_re + i * g
            h4 = o * torch.tanh(c4)
            
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.lstm_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 64, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c4_up = F.interpolate(c4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            c_re = self.reducec4(c4_up)
            c5 = f * c_re + i * g
            h5 = o * torch.tanh(c5)
            
            
        elif self.decoder == "GRU":
            
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv1(torch.cat([h0_up, x1], dim=1))
            cc_r, cc_z = torch.split(combined_conv, 2048, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv1(torch.cat([x1, r * h0_up], dim=1)))
            h1 = z * h0_up + (1 - z) * h_hat
            #h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv2(torch.cat([h1_up, x2], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [2048,1024], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv2(torch.cat([x2, r * h1_up], dim=1)))
            h1_re = self.reduceh1(h1_up)
            h2 = z * h1_re + (1 - z) * h_hat
            #h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv3(torch.cat([h2_up, x3], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [1024,512], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv3(torch.cat([x3, r * h2_up], dim=1)))
            h2_re = self.reduceh2(h2_up)
            h3 = z * h2_re + (1 - z) * h_hat
            #h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv4(torch.cat([h3_up, x4], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [512,256], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv4(torch.cat([x4, r * h3_up], dim=1)))
            h3_re = self.reduceh3(h3_up)
            h4 = z * h3_re + (1 - z) * h_hat
            #h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            combined_conv = self.gru_catconv5(torch.cat([h4_up, x5], dim=1))
            cc_r, cc_z = torch.split(combined_conv, [256,64], dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv5(torch.cat([x5, r * h4_up], dim=1)))
            h4_re = self.reduceh4(h4_up)
            h5 = z * h4_re + (1 - z) * h_hat
            #h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class
            
        elif self.decoder == "vanilla":
            fuse = torch.cat((x1, self._upsample(h0, x1)), 1)
            flow = self.flowconv1(fuse)
            h0 = self._flow_align_module(x1, h0, flow)
            h0_up = F.interpolate(h0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
            h1 = torch.relu(self.vanilla_conv1(torch.cat([h0_up, x1], dim=1)))
            #h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            fuse = torch.cat((x2, self._upsample(h1, x2)), 1)
            flow = self.flowconv2(fuse)
            h1 = self._flow_align_module(x2, h1, flow)
            h1_up = F.interpolate(h1, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
            h2 = torch.relu(self.vanilla_conv2(torch.cat([h1_up, x2], dim=1)))
            #h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            fuse = torch.cat((x3, self._upsample(h2, x3)), 1)
            flow = self.flowconv3(fuse)
            h2 = self._flow_align_module(x3, h2, flow)
            h2_up = F.interpolate(h2, size=[x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
            h3 = torch.relu(self.vanilla_conv3(torch.cat([h2_up, x3], dim=1)))
            #h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h3 = self._flow_align_module(x4, h3, flow)
            h3_up = F.interpolate(h3, size=[x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
            h4 = torch.relu(self.vanilla_conv4(torch.cat([h3_up, x4], dim=1)))
            #h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h4 = self._flow_align_module(x5, h4, flow)
            h4_up = F.interpolate(h4, size=[x5.size(2), x5.size(3)], mode='bilinear', align_corners=True)
            h5 = torch.relu(self.vanilla_conv5(torch.cat([h4_up, x5], dim=1)))
            #h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class
            
        x1, x2, x3, x4, x5 = h5, h4, h3, h2, h1
        h1, h2, h3, h4, hd5 = x1, x2, x3, x4, x5
        
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))#([1, 64, 22, 22])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))#([1, 64, 22, 22])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._upsample(hd5, h4))))
        #print(h1_PT_hd4.size(), h2_PT_hd4.size(), h3_PT_hd4.size(), h4_Cat_hd4.size(), hd5_UT_hd4.size())
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels
        
        #print(hd4.size())#torch.Size([1, 320, 22, 22])
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._upsample(hd4, h3))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._upsample(hd5, h3))))
        #print(h1_PT_hd3.size(), h2_PT_hd3.size(), h3_Cat_hd3.size(), hd4_UT_hd3.size(), hd5_UT_hd3.size())
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels
        #print(hd3.size())#torch.Size([1, 320, 45, 45])
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._upsample(hd3, h2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._upsample(hd4, h2))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._upsample(hd5, h2))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels
        #print(hd2.size())#torch.Size([1, 320, 90, 90])
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._upsample(hd2, h1))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._upsample(hd3, h1))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._upsample(hd4, h1))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._upsample(hd5, h1))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        #finalseg = hd1
        h1 = self.conv1_score_block(hd1)# 1/16,class torch.Size([1, 4, 11, 13])
        h2 = self.conv2_score_block(hd2)# 1/8,class torch.Size([1, 4, 22, 27])
        h3 = self.conv3_score_block(hd3)# 1/4,class torch.Size([1, 4, 45, 54])
        h4 = self.conv4_score_block(hd4)# 1/2,class torch.Size([1, 4, 90, 108])
        h5 = self.conv5_score_block(hd5)# 1,class torch.Size([1, 4, 181, 217])
        
        if self.decoder == "vanilla" or self.decoder == "GRU":        
            p1 = self.RDC(x_cur=h1, h_pre=h5);#print(p1.size())#torch.Size([1, 4, 181, 181])
            p2 = self.RDC(x_cur=h1, h_pre=h4);#print(p2.size())#torch.Size([1, 4, 181, 181])
            p3 = self.RDC(x_cur=h1, h_pre=h3);#print(p3.size())#torch.Size([1, 4, 181, 181])
            p4 = self.RDC(x_cur=h1, h_pre=h2);#print(p4.size())#torch.Size([1, 4, 181, 181])
        elif self.decoder == "LSTM": 
            c0 = self._init_cell_state(h5)
            p1, c1 = self.RDC(x_cur=h1, h_pre=h5, c_pre=c0);#print(p1.size())
            p2, c2 = self.RDC(x_cur=h1, h_pre=h4, c_pre=c1);#print(p1.size())
            p3, c3 = self.RDC(x_cur=h1, h_pre=h3, c_pre=c2);#print(p1.size())
            p4, c4 = self.RDC(x_cur=h1, h_pre=h2, c_pre=c3);#print(p1.size())
        
        out = torch.cat((h1, p1, p2, p3, p4), 1);#print(out.size())#torch.Size([1, 20, 181, 181])
        
        out = self.convp2(out);#print(out.size())
        out = self.relup2(self.bnp2(out));#print(out.size())
        out = self.convp3(out);#print(out.size())
        out = self._upsample(out, h1);#print(out.size())#torch.Size([1, 4, 181, 181])
        initseg = out
        # Direct Field
        #out = finalseg
        shift_n = 5
        for _ in range(shift_n):
            df = self.ConvDf_1x1(out)#torch.Size([1, 2, 181, 217])
            out, augdf = self.SelDF(out, df)#torch.Size([1, 64, 181, 217]), torch.Size([1, 2, 181, 217])

        select_x = self.fuse_conv(torch.cat([initseg, out], dim=1))
        out = select_x # torch.Size([1, 4, 181, 217])
        d1 = self.Conv_1x1(out)#torch.Size([1, 4, 181, 217])

        return [d1, df, initseg]

    def _init_cell_state(self, hidden_state):
        # return torch.zeros(hidden_state.size())

        return torch.zeros(hidden_state.size())



"""
Implementation code for miccai 2020.
"""
class UNetDSelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=4, auxseg=False):
        super(UNetDSelFuseFeature, self).__init__()
        
        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    )
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)
        

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0
        
        #print(df.shape) #torch.Size([1, 2, 217, 181])
        #visualize_flow(df)
        
        scale = 1.
        
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)

        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1)#.transpose(1, 2)
        grid_ = grid + 0.
        grid[...,0] = 2*grid_[..., 0] / (H-1) - 1
        grid[...,1] = 2*grid_[..., 1] / (W-1) - 1

        # features = []
        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))
        if self.auxseg:
            auxseg = self.auxseg_conv(x)
        else:
            auxseg = None

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return [select_x, auxseg]
        
class UNetD(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, bias=True, selfeat=False, shift_n=5, auxseg=False):
        super(UNetD, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.kernel_size= kernel_size
        self.bias = bias
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=input_channel,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        
        # Direct Field
        self.ConvDf_1x1 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = UNetDSelFuseFeature(64, auxseg=auxseg, shift_n=shift_n)

        self.Conv_1x1 = nn.Conv2d(64, self.n_classes, kernel_size=1, stride=1, padding=0)
        
    def _upsample(self, x, y, scale=1): # the size of x is as y
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')    
        
    
    def forward(self, inputs):
        x = inputs

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,self._upsample(d5, x4)),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,self._upsample(d4, x3)),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,self._upsample(d3, x2)),dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,self._upsample(d2, x1)),dim=1)
        d2 = self.Up_conv2(d2)

        # Direct Field
        df = self.ConvDf_1x1(d2)
        d2_auxseg = self.SelDF(d2, df)
        d2, auxseg = d2_auxseg[:2] #torch.Size([1, 64, 224, 224])
        d1 = self.Conv_1x1(d2)
        return [d1, df, auxseg]

"""
Implementation code for icassp 2020
"""

class UNet3(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, bias=True, is_deconv=True, is_batchnorm=True):
        super(UNet3, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.kernel_size= kernel_size
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
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
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

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
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

        '''stage 2d '''
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

        '''stage 1d'''
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

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        
    def _upsample(self, x, y, scale=1): # the size of x is as y
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')    
        
    
    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        h5 = self.conv5(h5)  # h5->20*20*1024
        hd5 = h5
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))#([1, 64, 40, 40])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))#([1, 64, 28, 23])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._upsample(hd5, h4))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._upsample(hd4, h3))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._upsample(hd5, h3))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._upsample(hd3, h2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._upsample(hd4, h2))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._upsample(hd5, h2))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._upsample(hd2, h1))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._upsample(hd3, h1))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._upsample(hd4, h1))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._upsample(hd5, h1))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels
        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return F.sigmoid(d1)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# resnetunet
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # outputs2 = F.interpolate(inputs2, size=[inputs1.size(2), inputs1.size(3)], mode='bilinear', align_corners=True)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear',
                                 align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))

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
            x = torch.tensor(x, dtype=torch.float32)
            x = conv(x)

        return x
        
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


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


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv(x)
        return x

def ResNet50DRNN(**kwargs):
    model = ResNetRNN(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# visualize flow field
def visualize_flow(flow):
    def cal_color(u, v):
        def colorwheel():
            colorbar = {'RY':15, 'YG':6, 'GC':4, 'CB':11, 'BM':13, 'MR':6}
            ncols = sum([v for k, v in colorbar.items()])
            colorwl = np.zeros(shape=[ncols, 3])
            col = 0
            for k, v in colorbar.items():
                if k == 'RY':
                    colorwl[col:v + col , 0] = 255
                    colorwl[col:v + col , 1] = np.array(np.floor(255 * np.linspace(0, v - 1,v) / v))
                if k == 'YG':
                    colorwl[col:v + col, 0] = 255 - np.array(np.floor(255 * np.linspace(0, v - 1, v) / v))
                    colorwl[col:v + col, 1] = 255
                if k == 'GC':
                    colorwl[col:v + col, 1] = 255
                    colorwl[col:v + col, 2] = np.array(np.floor(255 * np.linspace(0, v - 1, v) / v))
                if k == 'CB':
                    colorwl[col:v + col, 1] = 255 - np.array(np.floor(255 * np.linspace(0, v - 1, v) / v))
                    colorwl[col:v + col, 2] = 255
                if k == 'BM':
                    colorwl[col:v + col, 0] = np.array(np.floor(255 * np.linspace(0, v - 1, v) / v))
                    colorwl[col:v + col, 2] = 255
                if k == 'MR':
                    colorwl[col:v + col, 0] = 255
                    colorwl[col:v + col, 2] = 255 - np.array(np.floor(255 * np.linspace(0, v - 1, v) / v))
                col = col + v
            return  colorwl
        color_wl = colorwheel()
        n_cols = color_wl.shape[0]
        rad = np.sqrt(u**2 + v**2)
        t_a = np.arctan2(-v, -u) / np.pi
        fk = (t_a + 1) / 2 * (n_cols - 1)
        k0 = np.floor(fk).astype(np.int)
        k1 = k0 + 1
        k1 =  k1 * (k1 != n_cols)
        f = fk - k0
        img = np.zeros(shape=[u.shape[0], u.shape[1], 3])
        for i_color in range(color_wl.shape[1]):
            tmp = color_wl[:, i_color]
            col0 = np.zeros(shape=k0.shape)
            col1 = np.zeros(shape=k1.shape)
            for i in range(k0.shape[0]):
                for j in range(k0.shape[1]):
                    col0[i][j] = tmp[k0[i][j]] / 255
                    col1[i][j] = tmp[k1[i][j]] / 255
            col = (1 - f) * col0 + f * col1

            idx = (rad <= 1)
            for i in range(idx.shape[0]):
                for j in range(idx.shape[1]):
                    if idx[i][j]:
                        col[i][j] = 1 - rad[i][j] * (1 - col[i][j])
                    if not idx[i][j]:
                        col[i][j] = col[i][j]*0.75
            img[:, :, i_color] = np.floor(255 * col)
        return  img.astype(np.uint8)
    print(flow.shape)
    UNKNOWN_FLOW_TH = 1e9
    eps = 2.2204e-16
    if(flow.shape[1]!=2):
        flow = flow.permute(0, 3, 1, 2)
    bs = flow.shape[0]
    height = flow.shape[2]
    width = flow.shape[3]
    flow_u = flow[:, 0]
    flow_v = flow[:, 1]
    '''print('flow_u', file=f)
    print(flow_u, file=f)
    print('flow_v', file=f)
    print(flow_v, file=f)
    print('==================')'''
    #print(flow_u.shape, flow_v.shape)
    idxUnknown = (abs(flow_u) > UNKNOWN_FLOW_TH) * (abs(flow_v) > UNKNOWN_FLOW_TH)
    flow_u = flow_u * (~ idxUnknown)
    flow_v = flow_v * (~ idxUnknown)
    img_flow = []
    for i_b in range(bs):
        temp_u = flow_u[i_b].squeeze()
        temp_v = flow_v[i_b].squeeze()
        temp_u = temp_u.detach().numpy()
        temp_v = temp_v.detach().numpy()
        #max_u = temp_u.max()
        #max_v = temp_v.max()
        rad = np.sqrt(temp_u**2 + temp_v**2)
        max_rad = rad.max()

        temp_u = temp_u / (max_rad + eps)
        temp_v = temp_v / (max_rad + eps)
        
        temp_img = cal_color(temp_u, temp_v)
        temp_img = cal_color(temp_u, temp_v)
        img_flow.append(temp_img)
    visflow = np.array(img_flow)
    
    visflow = np.squeeze(visflow, axis=0)
    print(visflow.shape)
    scipy.misc.imsave(pjoin('./training_vis','{}.bmp'.format(height)),visflow)