from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x-mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
    def forward(self, x):
        return self.fn(self.norm(x))

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_inputkv = nn.Conv2d(dim, dim*2, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x, s=None):
        h, w = x.shape[-2:]
        heads = self.heads
        if s == None:
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        else:
            q, k, v = (self.to_q(x), *self.to_inputkv(s).chunk(2, dim=1))

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
    def forward(self, x):
        return self.net(x)


class TRD(nn.Module):
    def __init__(self, *, ch_in, ch_out, heads, ff_expansion, reduction_ratio, num_layers=2):
        # ff_expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), heads=(8, 5, 2, 1)
        super().__init__()
        block_kernel_stride_pad = (3, 2, 1)
        self.block = nn.ModuleList([])
        kernel = 3
        stride = 2
        padding = 1
        dim_in = ch_in
        dim_out = ch_out
        get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
        expand = nn.Linear(dim_in * (kernel ** 2), 4 * dim_in * (kernel ** 2), bias=False)
        overlap_patch_embed = nn.Conv2d(4 * dim_in * (kernel ** 2), dim_out, 1)
        layers_a = nn.ModuleList([])
        for _ in range(num_layers):
            layers_a.append(nn.ModuleList([
                PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                PreNorm(dim_out, MixFeedForward(dim=dim_in, expansion_factor=ff_expansion))
            ]))
        layers_b = nn.ModuleList([])
        for _ in range(num_layers):
            layers_b.append(nn.ModuleList([
                PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                PreNorm(dim_out, MixFeedForward(dim=dim_in, expansion_factor=ff_expansion))
            ]))
        layers_c = nn.ModuleList([])
        for _ in range(num_layers):
            layers_c.append(nn.ModuleList([
                PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                PreNorm(dim_out, MixFeedForward(dim=dim_in, expansion_factor=ff_expansion))
            ]))
        self.block.append(nn.ModuleList([
            get_overlap_patches,
            expand,
            overlap_patch_embed,
            layers_a,
            layers_b,
            layers_c
        ]))
    def forward(self, x, h):
        h_up = F.interpolate(h, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
        H, W = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, expand, overlap_embed, layersA, layersB, layersC) in self.block:
            x = get_overlap_patches(x)
            h_up = get_overlap_patches(h_up)
            print(h_up.size())
            print(x.size())
            # num_patches_x = x.shape[-1]
            # num_patches_h_up = h_up.shape[-1]
            # ratio_x = int(sqrt((H * W) / num_patches_x))
            # ratio_h_up = int(sqrt((H * W) / num_patches_h_up))
            x = expand(x)
            h_up = expand(h_up)
            x = rearrange(x, 'b c (h w) -> b c h w', h=H // 2)
            h_up = rearrange(h_up, 'b c (h w) -> b c h w', h=H // 2)
            x = overlap_embed(x)
            h_up = overlap_embed(h_up)
            _, C1, _, _ = x.shape
            _, C2, _, _ = h_up.shape
            x = rearrange(x, 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=2, p2=2, c=C1//4)
            h_up = rearrange(h_up, 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=2, p2=2, c=C2 // 4)
            for (attn, ff) in layersA:
                x = attn(x) + x
                x = ff(x) + x
            for (attn, ff) in layersB:
                h_up = attn(h_up) + h_up
                h_up = ff(x) + x
            for (attn, ff) in layersC:
                x = attn(x, h_up) + x
                x = ff(x) + x

        return x





class UTRD(nn.Module):
    def __init__(self, input_channel=3, n_classes=4, kernel_size=3, feature_scale=4, bias=True):

        super().__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downSampling
        self.conv1 = unetConv2(self.input_channel, filters[0], is_batchnorm=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm=True)

        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.TRD = TRD(ch_in=self.n_classes, ch_out=self.n_classes, heads=2, ff_expansion=8, reduction_ratio=8)
        # self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)

    def forward(self, input, cell_state=None):
        conv1 = self.conv1(input)  # 1,filters[0]

        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)  # 1/2,filters[1]

        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)  # 1/4,filters[2]

        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)  # 1/8,filters[3]

        maxpool4 = self.maxpool4(conv4)
        conv5 = self.center(maxpool4)  # 1/16,filters[4]

        x1 = self.score_block5(conv5)  # 1/16,class
        x2 = self.score_block4(conv4)  # 1/8,class
        x3 = self.score_block3(conv3)  # 1/4,class
        x4 = self.score_block2(conv2)  # 1/2,class
        x5 = self.score_block1(conv1)  # 1,class

        h1 = self.TRD(x2, x1)  # 1/16,class
        h2 = self.TRD(x3, h1)  # 1/8,class
        h3 = self.TRD(x4, h2)  # 1/4,class
        h4 = self.TRD(x5, h3)  # 1/2,class

        return h4

