import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UNetConv2d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetUP(nn.Module):
    def __init__(self, in_dim, out_dim, is_deconv=True):
        super().__init__()
        self.conv = UNetConv2d(in_dim, out_dim)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, y):
        y = self.up(y)
        # y = F.interpolate(y, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
        # offset = y.size()[2] - x.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # x = F.pad(x, padding)
        y = F.interpolate(y, size=[x.size(2), x.size(3)], mode='bilinear',
                          align_corners=True)
        output = self.conv(torch.cat([x, y], 1))
        return output

class RLACell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=False, cell_type='LSTM'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.cell_type = cell_type

        self.lstm_catconv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                      out_channels=4 * self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # cur_state is a tuple
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.lstm_catconv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class UNetRLA(nn.Module):
    def __init__(self, input_channel, n_classes, kernel_size, expansion=4, bias=True):
        super().__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = UNetConv2d(self.input_channel, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2d(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2d(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2d(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.bottom = UNetConv2d(filters[3], filters[4])




