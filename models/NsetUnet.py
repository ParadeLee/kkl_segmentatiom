import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# helpers
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class NesT(nn.Module):
    def __init__(
            self,
            x,
            image_size,
            patch_size,
            num_classes,
            dim,
            heads,
            num_hierarchies, # input
            block_repeats,
            mlp_mult = 4,
            channels = 3,
            dim_head = 64,
            dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimension must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2 # sequence length is held constant
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in hierarchies]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        for level, heads, (dim_in, dim_out), block_repeats in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level = 0
            depth = block_repeats
            self.layers.append(nn.Modulelist([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


