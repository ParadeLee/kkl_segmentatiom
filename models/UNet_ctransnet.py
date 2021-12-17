import torch
import torch.nn as nn
import torch.nn.functional as F
from models.UCtransBlock.UCtrans import ChannelTransformer


__all__ = ['UNet_Ctrans']


"""
Implementation code for UNet.
"""



class UNet_Ctrans(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=4, is_deconv=True, input_channel=3, is_batchnorm=True,
            vis=False, img_size=224
    ):
        super().__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        # print(filters)

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.mtc = ChannelTransformer(vis, img_size,
                                      channel_num=[filters[0], filters[1], filters[2], filters[3]]
                                      )
        # upsampling
        self.up_concat4 = UpBlock_attention(filters[4], filters[3], self.is_deconv, nb_Conv=2)
        self.up_concat3 = UpBlock_attention(filters[3], filters[2], self.is_deconv, nb_Conv=2)
        self.up_concat2 = UpBlock_attention(filters[2], filters[1], self.is_deconv, nb_Conv=2)
        self.up_concat1 = UpBlock_attention(filters[1], filters[0], self.is_deconv, nb_Conv=2)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        conv1, conv2, conv3, conv4, att_weights = self.mtc(conv1, conv2, conv3, conv4)
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        final = self.final(up1)
        if self.vis:  # visualize the attention maps
            return final, att_weights
        else:
            return final

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
        # outputs2 = F.interpolate(inputs2, size=[inputs1.size(2), inputs1.size(3)], mode='bilinear', align_corners=True)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv, nb_Conv, activation='ReLU'):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        up = F.interpolate(up, size=[skip_x_att.size(2), skip_x_att.size(3)], mode='bilinear',
                                 align_corners=True)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)



class unetUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp2, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))





# def debug_model():
#     model = UNet_Ctrans()
#     print(model)
#
# if __name__ == '__main__':
#     debug_model()