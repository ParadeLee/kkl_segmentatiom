import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class UNetConv2d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
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
        x = self.up(x)
        # y = F.interpolate(y, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
        # offset = y.size()[2] - x.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # x = F.pad(x, padding)
        x = F.interpolate(x, size=[y.size(2), y.size(3)], mode='bilinear',
                          align_corners=True)
        output = self.conv(torch.cat([x, y], 1))
        return output

class RDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='LSTM'):
        """
        Recurrent Decoding Cell (RDC) module.
        :param hidden_dim:
        :param kernel_size: conv kernel size
        :param bias: if or not to add a bias term
        :param decoder: <name> [options: 'vanilla, LSTM, GRU']
        """
        super(RDC, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.decoder = decoder
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size,
                                     padding=self.padding, bias=self.bias)
        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                  padding=self.padding, bias=self.bias)
        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size,
                                      padding=self.padding, bias=self.bias)
        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                      padding=self.padding, bias=self.bias)

    def forward(self, x_cur, h_pre, c_pre=None):
        if self.decoder == "LSTM":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
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
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.gru_catconv(combined)
            cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv(torch.cat([x_cur, r * h_pre_up], dim=1)))
            h_cur = z * h_pre_up + (1 - z) * h_hat

            return h_cur

        elif self.decoder == "vanilla":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.vanilla_conv(combined)
            h_cur = torch.relu(combined_conv)

            return h_cur

class RLACell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=False, cell_type='LSTM'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
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
        h_cur = F.interpolate(h_cur, size=[input_tensor.size(2), input_tensor.size(3)],
                                 mode='bilinear', align_corners=True)
        c_cur = F.interpolate(c_cur, size=[input_tensor.size(2), input_tensor.size(3)],
                              mode='bilinear', align_corners=True)
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

class RLA(nn.Module):
    expansion = 1
    def __init__(self, in_dim, out_dim, stride=1, rla_channel=16,
                 groups=1, base_width=64, dilation=1, reduction=16,
                 down_sample=None):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        width = int(out_dim * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_dim + rla_channel, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_dim * self.expansion)
        self.bn3 = norm_layer(out_dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = conv1x1(out_dim * self.expansion, rla_channel)
        self.bn_rla = norm_layer(rla_channel)
        self.tan_h = nn.Tanh()
        self.RLACell = RLACell(rla_channel, rla_channel)
        self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))


    def forward(self, x, h, c):
        identity = x
        x = torch.cat((x, h), dim=1)

        # 1*1block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3*3block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1*1block
        out = self.conv3(out)
        out = self.bn3(out)
        y = out
        # out += identity
        out = self.relu(out)

        # update RNN
        y = self.conv_out(y)
        y = self.bn_rla(y)
        y = self.tan_h(y)
        h, c = self.RLACell(y, (h, c))
        h = self.averagePooling(h)
        c = self.averagePooling(c)

        return out, h, c

class UNetRLARDC(nn.Module):
    def __init__(self, input_channel=3, n_classes=4, kernel_size=3, expansion=4, bias=True,
                 rla_channel=16, Decoder='LSTM'):
        super().__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.bias = bias
        self.rla_channel = rla_channel
        self.decoder = Decoder

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.expansion) for x in filters]

        # downsampling
        self.conv1 = UNetConv2d(self.input_channel, filters[0])
        self.proj1 = nn.Conv2d(filters[0]*2, filters[0], 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.RLA1 = RLA(self.input_channel, filters[0])

        self.conv2 = UNetConv2d(filters[0], filters[1])
        self.proj2 = nn.Conv2d(filters[1] * 2, filters[1], 1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.RLA2 = RLA(filters[0], filters[1])

        self.conv3 = UNetConv2d(filters[1], filters[2])
        self.proj3 = nn.Conv2d(filters[2] * 2, filters[2], 1, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.RLA3 = RLA(filters[1], filters[2])

        self.conv4 = UNetConv2d(filters[2], filters[3])
        self.proj4 = nn.Conv2d(filters[3] * 2, filters[3], 1, 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.RLA4 = RLA(filters[2], filters[3])

        self.bottom = UNetConv2d(filters[3], filters[4])
        self.RLA5 = RLA(filters[3], filters[4])

        # skip connection
        self.score_block1 = nn.Sequential(
            nn.Conv2d(filters[0]*2, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1]*2, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2]*2, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3]*2, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4]*2, self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)

        # self.up1 = UNetUP(filters[4], filters[3])
        # self.up2 = UNetUP(filters[3], filters[2])
        # self.up3 = UNetUP(filters[2], filters[1])
        # self.up4 = UNetUP(filters[1], filters[0])

        self.seg_head = nn.Sequential(
            nn.Conv2d(filters[0], self.n_classes, 3, 1, 1)
        )
    def _init_cell_state(self, tensor):
        # return torch.zeros(tensor.size()).cuda(0)
        return torch.zeros(tensor.size())


    def forward(self, x):
        B, _, H, W = x.size()
        h = torch.zeros(B, self.rla_channel, H, W)
        c = torch.zeros(B, self.rla_channel, H, W)

        # stage1
        x_down1 = self.conv1(x)
        x_res1, h1, c1 = self.RLA1(x, h, c)
        # x_down1_skip = x_down1 + x_res1
        x_down1_skip = torch.cat([x_down1, x_res1], dim=1)
        x_down1 = self.proj1(x_down1_skip)
        x_down1 = self.maxpool1(x_down1)

        # stage2
        x_down2 = self.conv2(x_down1)
        x_res2, h2, c2 = self.RLA2(x_down1, h1, c1)
        # x_down2_skip = x_down2 + x_res2
        x_down2_skip = torch.cat([x_down2, x_res2], dim=1)
        x_down2 = self.proj2(x_down2_skip)
        x_down2 = self.maxpool1(x_down2)

        # stage3
        x_down3 = self.conv3(x_down2)
        x_res3, h3, c3 = self.RLA3(x_down2, h2, c2)
        # x_down3_skip = x_down3 + x_res3
        x_down3_skip = torch.cat([x_down3, x_res3], dim=1)
        x_down3 = self.proj3(x_down3_skip)
        x_down3 = self.maxpool1(x_down3)

        # stage4
        x_down4 = self.conv4(x_down3)
        x_res4, h4, c4 = self.RLA4(x_down3, h3, c3)
        # x_down4_skip = x_down4 + x_res4
        x_down4_skip = torch.cat([x_down4, x_res4], dim=1)
        x_down4 = self.proj4(x_down4_skip)
        x_down4 = self.maxpool1(x_down4)

        # stage bottom
        x_bottom = self.bottom(x_down4)
        x_res5, h5, c5 = self.RLA5(x_down4, h4, c4)
        # x_up0 = x_bottom + x_res5
        x_up0 = torch.cat([x_bottom, x_res5], dim=1)

        x1 = self.score_block5(x_up0)  # 1/16,class
        x2 = self.score_block4(x_down4_skip)  # 1/8,class
        x3 = self.score_block3(x_down3_skip)  # 1/4,class
        x4 = self.score_block2(x_down2_skip)  # 1/2,class
        x5 = self.score_block1(x_down1_skip)  # 1,class

        h0 = self._init_cell_state(x1)
        c0 = self._init_cell_state(h0)

        h1, c1 = self.RDC(x_cur=x1, h_pre=h0, c_pre=c0)  # 1/16,class
        h2, c2 = self.RDC(x_cur=x2, h_pre=h1, c_pre=c1)  # 1/8,class
        h3, c3 = self.RDC(x_cur=x3, h_pre=h2, c_pre=c2)  # 1/4,class
        h4, c4 = self.RDC(x_cur=x4, h_pre=h3, c_pre=c3)  # 1/2,class
        h5, c5 = self.RDC(x_cur=x5, h_pre=h4, c_pre=c4)  # 1,class

        out = h5

        # # Decoder
        # x_up1 = self.up1(x_up0, x_down4_skip)  # 28 * 28
        # x_up2 = self.up2(x_up1, x_down3_skip)  # 56 * 56
        # x_up3 = self.up3(x_up2, x_down2_skip)  # 112 * 112
        # x_up4 = self.up4(x_up3, x_down1_skip)
        #
        # out = self.seg_head(x_up4)

        return out












