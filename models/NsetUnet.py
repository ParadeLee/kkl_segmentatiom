import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# helpers
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.): # Transformer的depth代表了Transformer_block的深度
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# classes
class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class NesTUnet(nn.Module):
    def __init__(
            self,
            *,
            image_size, # 224*224
            patch_size, # 4
            num_classes, # 4
            dim, # dim=96
            heads, # num_heads=3
            num_hierarchies, # input_hierarchies = 3
            block_repeats, # block_repeats=(2, 2, 8)
            mlp_mult=4,
            channels=3,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, \
            'Image dimension must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2 # sequence length is held constant
        hierarchies = list(reversed(range(num_hierarchies))) # [2, 1, 0]
        hierarchies_up = list(range(num_hierarchies)) # [0, 1, 2]
        mults = [2 ** i for i in hierarchies] # [4, 2, 1]
        # mults = [4 for i in hierarchies]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults)) # [384， 192， 98]

        layer_dims = [*layer_dims, layer_dims[-1]] # [384， 192， 98， 98]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:]) # [(4*dim, 2*dim), (2*dim, 1*dim), (1*dim, 1*dim)]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )
        # 构建Encoder层
        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeats in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeats
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # 构建Decoder层
        # self.conv_bottom = DoubleConv(96, 192)
        self.expand1 = nn.ConvTranspose2d(96, 96, 2, stride=2)
        self.expand2 = nn.ConvTranspose2d(192, 192, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.conv1 = DoubleConv(192, 192)
        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.conv2 = DoubleConv(192, 192)
        self.up3 = nn.ConvTranspose2d(192, 192, 2, stride=2)

        self.PatchExpand = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 2, stride=2),
            nn.ConvTranspose2d(192, 96, 2, stride=2)
        )
        self.conv_final =DoubleConv(96, num_classes)

        # self.mlp_head = nn.Sequential(
        #     LayerNorm(dim),
        #     Reduce('b c h w -> b c', 'mean'),
        #     nn.Linear(dim, num_classes)
        # )
        # self.mlp_head = nn.Sequential(
        #     nn.Conv2d(32, 1, 1),
        #     nn.BatchNorm2d(num_classes),
        #     nn.ReLU(inplace=True)
        # )
        self.bottom_head = nn.Sequential(
            LayerNorm(dim),
            # Reduce('b c h w -> b c', 'mean'),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)
        skip_connection = []

        # Encoder部分

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
            skip_connection.append(x)

        # Decoder部分
        x = self.bottom_head(x)

        x = self.up1(x)
        x = torch.cat([x, skip_connection[2]], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        skip_connection[1] = self.expand1(skip_connection[1])
        x = torch.cat([x, skip_connection[1]], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        skip_connection[0] = self.expand2(skip_connection[0])
        x = torch.cat([x, skip_connection[0]], dim=1)
        x = self.PatchExpand(x)
        x = self.conv_final(x)
        out = torch.relu(x)

        return out
        # return self.mlp_head(x)
