import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):  # 卷积BN和激活函数
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)
        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, configs, axial=False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = int(self.dim / configs["head"])
        self.all_head_size = self.num_head * self.attention_head_size

        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)

        self.out = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)  # 在变量前加*号代表解包的意思，new_x_shape是一个元组
        return x

    def forward(self, x):
        # first row and col attention
        if self.axial:
            # row attention (single head attention)
            b, h, w, c = x.shape
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer_x = mixed_query_layer.view(b * h, w, -1)
            key_layer_x = mixed_key_layer.view(b * h, w, -1).transpose(-1, -2)
            attention_scores_x = torch.matmul(query_layer_x,
                                              key_layer_x)  # (b*h, w, w, c)
            attention_scores_x = attention_scores_x.view(b, -1, w,
                                                         w)  # (b, h, w, w)

            # col attention  (single head attention)
            query_layer_y = mixed_query_layer.permute(0, 2, 1,
                                                      3).contiguous().view(
                                                          b * w, h, -1)
            key_layer_y = mixed_key_layer.permute(
                0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attention_scores_y = torch.matmul(query_layer_y,
                                              key_layer_y)  # (b*w, h, h, c)
            attention_scores_y = attention_scores_y.view(b, -1, h,
                                                         h)  # (b, w, h, h)

            return attention_scores_x, attention_scores_y, mixed_value_layer

        else:

            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer = self.transpose_for_scores(mixed_query_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()  # (b, p, p, head, n, c)
            key_layer = self.transpose_for_scores(mixed_key_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            value_layer = self.transpose_for_scores(mixed_value_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()

            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            atten_probs = self.softmax(attention_scores)

            context_layer = torch.matmul(
                atten_probs, value_layer)  # (b, p, p, head, win, h)
            context_layer = context_layer.permute(0, 1, 2, 4, 3,
                                                  5).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.all_head_size, )
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)

        return attention_output

class WinAttention(nn.Module):
    def __init__(self, configs, dim):
        super(WinAttention, self).__init__()
        self.window_size = configs["win_size"]
        self.attention = Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            right_size = h + self.window_size - h % self.window_size
            new_x = torch.zeros((b, c, right_size, right_size))
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:,
                  x.shape[3]:] = x[:, :, (x.shape[2] - right_size):,
                                   (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5,
                      1).contiguous().view(b, h // self.window_size,
                                           w // self.window_size,
                                           self.window_size * self.window_size,
                                           c).cpu()
        x = self.attention(x)  # (b, p, p, win, h)
        return x


class DlightConv(nn.Module):
    def __init__(self, dim, configs):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)  # (b, n, n, 1, h)
        x_prob = self.softmax(self.linear(avg_x))  # (b, n, n, win)

        x = torch.mul(h,
                      x_prob.unsqueeze(-1))  # (b, p, p, 16, h) (b, p, p, 16)
        x = torch.sum(x, dim=-2)  # (b, n, n, 1, h)
        return x


class GaussianTrans(nn.Module):
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x_full, atten_y_full, value_full = x  # atten_x_full(b, h, w, w, c)   atten_y_full(b, w, h, h, c) value_full(b, h, w, c)
        new_value_full = torch.zeros_like(value_full)

        for r in range(x.shape[1]):  # row
            for c in range(x.shape[2]):  # col
                atten_x = atten_x_full[:, r, c, :]  # (b, w)
                atten_y = atten_y_full[:, c, r, :]  # (b, h)

                dis_x = torch.tensor([(h - c)**2 for h in range(x.shape[2])
                                      ]).cpu()  # (b, w)
                dis_y = torch.tensor([(w - r)**2 for w in range(x.shape[1])
                                      ]).cpu()  # (b, h)

                dis_x = -(self.shift * dis_x + self.bias).cpu()
                dis_y = -(self.shift * dis_y + self.bias).cpu()

                atten_x = self.softmax(dis_x + atten_x)
                atten_y = self.softmax(dis_y + atten_y)

                new_value_full[:, r, c, :] = torch.sum(
                    atten_x.unsqueeze(dim=-1) * value_full[:, r, :, :] +
                    atten_y.unsqueeze(dim=-1) * value_full[:, :, c, :],
                    dim=-2)
        return new_value_full

class CSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.win_atten = WinAttention(configs, dim)
        self.dlightconv = DlightConv(dim, configs)
        self.global_atten = Attention(dim, configs, axial=True)
        self.gaussiantrans = GaussianTrans()
        #self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        #self.maxpool = nn.MaxPool2d(2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        '''

        :param x: size(b, n, c)
        :return:
        '''
        origin_size = x.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
            origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        x = self.win_atten(x)  # (b, p, p, win, h)
        b, p, p, win, c = x.shape
        h = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                   c).permute(0, 1, 3, 2, 4, 5).contiguous()
        h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                   c).permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        x = self.dlightconv(x)  # (b, n, n, h)
        atten_x, atten_y, mixed_value = self.global_atten(
            x)  # (atten_x, atten_y, value)
        gaussian_input = (x, atten_x, atten_y, mixed_value)
        x = self.gaussiantrans(gaussian_input)  # (b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.up(x)
        x = self.queeze(torch.cat((x, h), dim=1)).permute(0, 2, 3,
                                                          1).contiguous()
        x = x[:, :origin_h, :origin_w, :].contiguous()
        x = x.view(b, -1, c)

        return x

class MEAttention(nn.Module):
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)

        return x

class att_modul(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x
        x = self.layerNorm(x)
        x = self.CSAttention(x)
        x = h + x
        h = x
        x = self.layerNorm(x)
        x = self.EAttention(x)
        x = h + x

        return x

class EncoderStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)  # out_dim, model_dim
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))



class newtrans(nn.Module):
    def __init__(self, out_ch=4):
        super().__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential()



configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}