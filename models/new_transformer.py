from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)


    def forward(self, x, x_all, mask_all=None):
        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))




# helps

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

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
        mean = torch.mean(x, dim = 1, keepdim=True)
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

class Attention(nn.Module):
    # def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
    def __init__(self, dim, num_heads=5, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


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

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class Enhance_MixFeedForward(nn.Module): # 1层
    def __init__(
        self,
        *,
        dim,
        expansion_factor  # 膨胀系数
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.layers = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, 1),
            DoubleConv(hidden_dim, hidden_dim, 3, padding=1),
            LayerNorm(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1)
        ])

        # self.conv1 = nn.Conv2d(dim, hidden_dim, 1),
        # self.double_conv = DoubleConv(hidden_dim, hidden_dim, 3, padding=1),
        # self.LN = LayerNorm(hidden_dim)
        # self.conv2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        res = x = self.layers[0](x)
        x = self.layers[1](x)
        x = torch.add(res, x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        # x = self.conv1(x)
        # res = x
        # x = self.double_conv(x)
        # x = torch.add(res, x)
        # x = self.LN(x)
        # x = self.conv2(x)
        return x


# class PatchMerging(nn.Module):
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super(self).__init__()

#  上采样
class PatchExpand(nn.Module):
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

        return x



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
            # patch merging
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x
            layer_connected.append(x)

        return layer_connected

class UTDecoder(nn.Module):
    def __init__(
            self,
            dims,
            heads,
            ff_expansion,
            reduction_ratio,  # (8, 4, 2, 1)
            num_connected_layers, # 4
            decoder_dim,
            num_classes,
    ):
        super().__init__()
        # layer_connection
        proj_dim = dims  # (32, 64, 160, 256)
        fix_dims = 160
        self.proj_norm = LayerNorm(fix_dims)
        # self.LN = LayerNorm()
        self.connected_proj1 = nn.Sequential(
            projection(proj_dim[0], fix_dims),
            nn.ReLU(inplace=True),
        )
        self.connected_proj2 = nn.Sequential(
            projection(proj_dim[1], fix_dims),
            nn.ReLU(inplace=True),
        )
        self.connected_proj3 = nn.Sequential(
            projection(proj_dim[2], fix_dims),
            nn.ReLU(inplace=True),
        )
        self.connected_proj4 = nn.Sequential(
            projection(proj_dim[3], fix_dims),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_connected_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(fix_dims, Attention(dim=fix_dims)),
                PreNorm(fix_dims, Enhance_MixFeedForward(dim=fix_dims, expansion_factor=ff_expansion[_])),
            ]))

        final_dims = ()
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(final_dims, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i)
        ) for i, final_dims in enumerate(final_dims)])

        # MLP层输出
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):  # 传入encoder的输出
        token = x
        token[0] = self.connected_proj1(token[0])
        token[1] = self.connected_proj2(token[1])
        token[2] = self.connected_proj3(token[2])
        token[3] = self.connected_proj4(token[3])

        merge_token = torch.cat((token[0], token[1], token[2], token[3]), dim=1)
        for (attn, ff) in self.layers:
            res_token = self.proj_norm(attn(self.proj_norm(merge_token),  ) + merge_token)
            split_token = torch.split(res_token, (160, 160, 160, 160), dim=1)
            ffn1 = ff(split_token[0])
            ffn2 = ff(split_token[1])
            ffn3 = ff(split_token[2])
            ffn4 = ff(split_token[3])
            merge_token = torch.cat((ffn1, ffn2, ffn3, ffn4), dim=1) + res_token
        decoder_token = torch.split(merge_token, (160, 160, 160, 160), dim=1)

        output = [torch.cat((a, b), dim=1) for a, b in zip(decoder_token, x)]
        fused = [to_fused(output) for output, to_fused in zip(output, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)


class Enhance_segfomer(nn.Module):
    def __init__(
            self,
            *,
            dims=(32, 64, 160, 256),
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=2,
            channels=3,
            decoder_dim=256,
            num_classes=4,
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = \
            map(partial(cast_tuple, depth=4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), \
            'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.encoder = UTEncoder(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
        )

        self.decoder = UTDecoder(
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,  # (8, 4, 2, 1)
            num_connected_layers=4,  # 4
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output




