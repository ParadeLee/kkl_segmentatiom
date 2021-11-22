import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride,
                              # padding=(patch_size[0] // 2, patch_size[1] // 2)
                              )
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
