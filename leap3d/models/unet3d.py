""" This code is adapted from the 2D UNet implementation in LEAP, which adapts from the repo https://github.com/milesial/Pytorch-UNet"""

import logging
import torch
from torch import nn
from torch.nn import functional as F


class Double3DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.LeakyReLU ,**kwargs):
        super(Double3DConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(mid_channels),
            activation(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, **kwargs):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            Double3DConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, activation=nn.LeakyReLU, **kwargs):
        super(Up3D, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Double3DConv(in_channels, out_channels, in_channels // 2, activation=activation)
        else:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Double3DConv(in_channels, out_channels, activation=activation, padding_mode='replicate')

    def forward(self, x1, x2):
        # x1 - previous layer output
        # x2 - skip connection output
        x1 = self.upsample(x1)
        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, n_conv=16, depth=4, fcn_core_layers=0, extra_params_number: int=0, bilinear=False, activation=nn.LeakyReLU, **kwargs):
        super(UNet3D, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_conv = n_conv
        self.bilinear = bilinear
        self.depth = depth
        self.fcn_core_layers = kwargs.get('fcn_core_layers', fcn_core_layers)
        self.extra_params_number = kwargs.get('extra_params_number', extra_params_number)

        self.inc = Double3DConv(input_dimension, n_conv)

        self.down1 = Down3D(n_conv, n_conv*2, activation=activation)
        self.down2 = Down3D(n_conv*2, n_conv*4, activation=activation)
        self.down3 = Down3D(n_conv*4, n_conv*8, activation=activation)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(n_conv*8, n_conv*16 // factor, activation=activation)

        # Inject extra parameters into the UNet using a FCN
        self.fcn_core = None
        if self.fcn_core_layers > 0:
            modules = []
            x_elements_number = (64 // (2**self.depth))**2 * (16 // (2**self.depth))
            for _ in range(self.fcn_core_layers):
                output_features = n_conv*(2**self.depth) * x_elements_number
                modules.append(nn.Linear(output_features + extra_params_number, output_features))
                modules.append(activation())
            self.fcn_core = nn.Sequential(*modules)

        self.up1 = Up3D(n_conv*16, n_conv*8 // factor, bilinear, activation=activation)
        self.up2 = Up3D(n_conv*8, n_conv*4 // factor, bilinear, activation=activation)
        self.up3 = Up3D(n_conv*4, n_conv*2 // factor, bilinear, activation=activation)
        self.up4 = Up3D(n_conv*2, n_conv, bilinear, activation=activation)
        self.outc = OutConv3D(n_conv, output_dimension)

        print(self.fcn_core_layers, self.extra_params_number)

    def forward(self, x, extra_params=None):
        in_shape = x.shape
        if len(in_shape) == 4:
            x = torch.unsqueeze(x, 0)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.fcn_core is not None:
            # print(x5.shape, extra_params.shape)
            shape = x5.shape
            x5 = x5.reshape(shape[0], -1)
            if self.extra_params_number != 0:
                x5 = torch.cat((x5, extra_params), dim=-1)
            x5 = self.fcn_core(x5)
            x5 = x5.reshape(shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        if len(in_shape) == 4:
            assert out.shape[0] == 1
            out = out[0]
        return out
