""" This code is adapted from the 2D UNet implementation in LEAP, which adapts from the repo https://github.com/milesial/Pytorch-UNet"""

import logging
import torch
from torch import nn
from torch.nn import functional as F


class Double2DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kwargs):
        super(Double2DConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down2D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Double2DConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up2D, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Double2DConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Double2DConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 - previous layer output
        # x2 - skip connection output
        x1 = self.upsample(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def put_channels_first(tensor):
    if len(tensor.shape) == 3:
        return tensor.permute(2, 0, 1)
    return tensor.permute(0, 3, 1, 2)

def put_channels_last(tensor):
    if len(tensor.shape) == 3:
        return tensor.permute(1, 2, 0)
    return tensor.permute(0, 2, 3, 1)


class UNet2D(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, n_conv=16, depth=4, bilinear=False):
        super(UNet2D, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_conv = n_conv
        self.bilinear = bilinear

        self.inc = Double2DConv(input_dimension, n_conv)

        self.down1 = Down2D(n_conv, n_conv*2)
        self.down2 = Down2D(n_conv*2, n_conv*4)
        self.down3 = Down2D(n_conv*4, n_conv*8)
        factor = 2 if bilinear else 1
        self.down4 = Down2D(n_conv*8, n_conv*16 // factor)
        self.up1 = Up2D(n_conv*16, n_conv*8 // factor, bilinear)
        self.up2 = Up2D(n_conv*8, n_conv*4 // factor, bilinear)
        self.up3 = Up2D(n_conv*4, n_conv*2 // factor, bilinear)
        self.up4 = Up2D(n_conv*2, n_conv, bilinear)
        self.outc = OutConv2D(n_conv, output_dimension)

    def forward(self, x):
        in_shape = x.shape
        if len(in_shape) == 3:
            x = torch.unsqueeze(x, 0)

        # TODO: Form data to have channel first
        x = put_channels_first(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = put_channels_last(out)
        if len(in_shape) == 3:
            assert out.shape[0] == 1
            out = out[0]
        return out
