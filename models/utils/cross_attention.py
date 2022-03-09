
import torch
import torch.nn as nn
import torch.nn.functional as F



def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.ReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class NonLocalAttention(nn.Module):
    def __init__(self, channel, fusionchannel, reduction=2, conv=default_conv):
        super().__init__()
        self.conv_q = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.ReLU())
        self.conv_k = BasicBlock(conv, fusionchannel, channel // reduction, 1, bn=False, act=nn.ReLU())
        self.conv_assembly = BasicBlock(conv, fusionchannel, channel, 1, bn=False, act=nn.ReLU())
        self.conv = BasicBlock(conv, channel*2, channel, 1, bn=False, act=nn.ReLU())

    def forward(self, x, fusionmap):
        x_embed_1 = self.conv_q(x)
        x_embed_2 = self.conv_k(fusionmap)
        x_assembly = self.conv_assembly(fusionmap)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        x_final = x_final.permute(0, 2, 1).view(N, -1, H, W)
        x_final = self.conv(torch.cat([x, x_final], dim=1))
        return x_final

