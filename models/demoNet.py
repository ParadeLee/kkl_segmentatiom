import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0]