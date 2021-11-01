from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helps

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes
# class MLP(nn.Module):
#     def __int__(self, input_size, out_size):
#         super().__int__()
#         self.linear = nn.Sequential(
#             nn.Linear(input_size, input_size // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_size // 2, input_size // 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_size // 4, out_size),
#         )
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out

class projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.proj(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class DoubleConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio # 减速比
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class Enhance_MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor  # 膨胀系数
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1),
        self.double_conv = DoubleConv(hidden_dim, hidden_dim, 3, padding=1),
        self.LN = LayerNorm(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        res = x = self.conv1(x)
        x = self.double_conv(x)
        x = torch.add(res, x)
        x = self.LN(x)
        x = self.conv2(x)
        return x


# class PatchMerging(nn.Module):
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super(self).__init__()

#  上采样
class PatchExpand(nn.module):
    def __init__(
            self,
            input_resolution, # [img_size[0]//patch_size[0], img_size[1]//patch_size[1]]
            dim,
            dim_scale=2,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expend(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)


class UTEncoder(nn.Module):
    def __init__(
            self,
            *,
            channels, # 3
            dims, # (32, 64, 160, 256)
            heads,# (1, 2, 5, 8),
            ff_expansion, # (8, 8, 4, 4),膨胀系数
            reduction_ratio,  # (8, 4, 2, 1)
            num_layers,
    ):
        super().__init__()
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))  # (3, 32, 64, 160, 256)
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        self.stages = nn.ModuleList([])
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in \
                zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, Enhance_MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))
            
    def forward(self, x):
        h, w = x.shape[-2:]
        layer_connected = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]

            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h//ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x
            layer_connected.append(x)

        return x, layer_connected

class UTDecoder(nn.Module):
    def __init__(
            self,
            dims,
            heads,
            ff_expansion,
            reduction_ratio,  # (8, 4, 2, 1)
            num_layers,
            layer_connected,
    ):
        super().__init__()
        # layer_connection
        self.layer_connected = layer_connected
        proj_dim = list(zip(32, 64, 160, 256))


        self.layers = nn.ModuleList([])
        dims = 256
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dims, EfficientSelfAttention(dim=dims, heads=heads, reduction_ratio=reduction_ratio)),
                PreNorm(dims, Enhance_MixFeedForward(dim=dims, expansion_factor=ff_expansion)),
            ]))


    def forward(self, x):
        token = []
        token = self.layer_connected


        return x


















class kiuT(nn.Module):
    def __init__(
            self,
            *,
            channels,
            dims, # (32, 64, 160, 256)
            heads,
            ff_expansion,
            reduction_ratio,
            # num_layers = 2
    ):
        super().__init__()
        dims = (channels, *dims) # (3, 32, 64, 160, 256)
        dim_in, dim_out = []
        dim_in = dims[:-1]
        dim_out = dims[1:]
        kernel = [7, 3, 3, 3]
        self.OverlapPatchEmbedding = nn.Sequential(
            nn.Unfold(kernel[0], stride=4, padding=3),
            nn.Conv2d(dim_in[0] * kernel[0] ** 2, dim_out[0], 1),
        )
        # get_overlap_patches = nn.Unfold(kernel[0], stride=4, padding=3)
        # overlap_patch_embed = nn.Conv2d(dim_in[0] * kernel[0] ** 2, dim_out[0], 1)
        self.Attention1 = nn.Sequential(
            PreNorm(dim_out[0], EfficientSelfAttention(dim=dim_out[0], heads= heads, reduction_ratio=reduction_ratio)),
            PreNorm(dim_out[0], Enhance_MixFeedForward(dim=dim_out[0], expansion_factor=ff_expansion)),
        )


