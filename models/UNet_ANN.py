import torch
import torch.nn as nn
import torch.nn.functional as F

class
     没用
__all__ = ['UNet_ANN']


"""
Implementation code for UNet.
"""
class conv_1(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)
        return output

class conv_2(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, 3, 2, 1)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)
        return output

class convup(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False):
        super().__init__()
        self.conv = conv_1(in_size + out_size, out_size)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
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



class UNet_ANN(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=3, is_deconv=True, input_channel=3, is_batchnorm=True
    ):
        super().__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # AAN
        self.out_dim = 1

        filters_ann = [8, 16, 32, 64]

        self.dim_layer_ann = conv_1(input_channel, filters_ann[0])
        self.conv1_aan = conv_2(filters_ann[0], filters_ann[1])
        self.conv2_aan = conv_2(filters_ann[1], filters_ann[2])
        self.conv3_aan = conv_2(filters_ann[2], filters_ann[3])

        self.bottom_aan = conv_1(filters_ann[3], filters_ann[3])
        self.up1_ann = convup(filters_ann[3], filters_ann[2])
        self.up2_ann = convup(filters_ann[2], filters_ann[1])
        self.up3_ann = convup(filters_ann[1], filters_ann[0])

        self.final_ann = conv_1(filters_ann[0], 1)

        # downsampling
        self.conv1 = unetConv2(self.input_channel + 1, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, gray):
        x = self.dim_layer_ann(inputs)
        conv1 = self.conv1_aan(x)
        conv2 = self.conv2_aan(conv1)
        conv3 = self.conv3_aan(conv2)
        up0 = self.bottom_aan(conv3)
        up1 = self.up1_ann(conv2, up0)
        up2 = self.up2_ann(conv1, up1)
        up3 = self.up3_ann(x, up2)
        out = self.final_ann(up3)
        y = out
        grad = torch.cat([y, gray], dim=1)  # 加入灰度图像，计算loss

        inputs = torch.cat([inputs, y], dim=1)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final, grad

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

