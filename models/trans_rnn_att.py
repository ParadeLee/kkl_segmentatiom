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

class AttRDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='vanilla'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernal_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.decoder = decoder
        self.bias = bias
        # self.n_classes = dims

        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size, padding=self.padding,
                                     bias=self.bias)

        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                  padding=self.padding, bias=self.bias)

        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size, padding=self.padding,
                                      bias=self.bias)

        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                      padding=self.padding, bias=self.bias)


    def forward(self, x_cur, h_pre, c_pre=None):
        c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

