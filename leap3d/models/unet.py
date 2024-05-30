""" This code is adapted from the 2D UNet implementation in LEAP, which adapts from the repo https://github.com/milesial/Pytorch-UNet"""

import logging
import torch
from torch import nn
from torch.nn import functional as F

USE_BIAS = False

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=bias),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True) if activation == nn.LeakyReLU else activation(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=bias),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True) if activation == nn.LeakyReLU else activation(),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3d(DoubleConv):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(DoubleConv3d, self).__init__(in_channels, out_channels, mid_channels=mid_channels, activation=activation, bias=bias, **kwargs)
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=bias),
            nn.BatchNorm3d(mid_channels),
            activation(inplace=True) if activation == nn.LeakyReLU else activation(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=bias),
            nn.BatchNorm3d(out_channels),
            activation(inplace=True) if activation == nn.LeakyReLU else activation(),
        )



class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3d(Down):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(Down3d, self).__init__(in_channels, out_channels, activation=activation, bias=bias, **kwargs)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels, activation=activation, bias=bias)
        )


class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            mid_channels = in_channels // 2
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            mid_channels = None

        self.conv = DoubleConv(in_channels, out_channels, mid_channels,
                               activation=activation, padding_mode='replicate', bias=bias)

    def get_input_padding(self, x1, x2):
        x1_size = x1.size()
        x2_size = x2.size()

        diff_x = x2_size[3] - x1_size[3]
        diff_y = x2_size[2] - x1_size[2]

        padding = [diff_x // 2, diff_x - diff_x // 2,
                   diff_y // 2, diff_y - diff_y // 2]

        return padding

    def forward(self, x1, x2):
        # x1 - previous layer output
        # x2 - skip connection output
        x1 = self.upsample(x1)
        # input is CHW
        padding = self.get_input_padding(x1, x2)

        x1 = F.pad(x1, padding)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3d(Up):
    def __init__(self, in_channels, out_channels, bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(Up3d, self).__init__(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias, **kwargs)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            mid_channels = in_channels // 2
        else:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            mid_channels = None

        self.conv = DoubleConv3d(in_channels, out_channels, mid_channels,
                                 activation=activation, padding_mode='replicate', bias=bias)

    def get_input_padding(self, x1, x2):
        x1_size = x1.size()
        x2_size = x2.size()

        diff_x = x2_size[3] - x1_size[3]
        diff_y = x2_size[2] - x1_size[2]
        diff_z = x2_size[4] - x1_size[4]

        padding = [diff_x // 2, diff_x - diff_x // 2,
                   diff_y // 2, diff_y - diff_y // 2,
                   diff_z // 2, diff_z - diff_z // 2]

        return padding


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='replicate', bias=bias)

    def forward(self, x):
        return self.conv(x)


class OutConv3d(OutConv):
    def __init__(self, in_channels, out_channels, bias=False):
        super(OutConv3d, self).__init__(in_channels, out_channels, bias=bias)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding_mode='replicate', bias=bias)


class UNet(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, n_conv=16, depth=4, bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_conv = n_conv
        self.bilinear = bilinear
        self.depth = depth

        self.inc = DoubleConv(input_dimension, n_conv, activation=activation, bias=bias)
        self.down = self.get_down_layers(n_conv, depth, activation=activation, bias=bias, bilinear=bilinear)
        self.up = self.get_up_layers(n_conv, depth, activation=activation, bias=bias, bilinear=bilinear)
        self.outc = OutConv(n_conv, output_dimension, bias=bias)

    def get_down_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        for i in range(depth - 1):
            in_channels = n_conv * (2**i)
            out_channels = in_channels * 2
            layers.append(Down(in_channels, out_channels, activation=activation, bias=bias))

        in_channels = n_conv * (2**(depth - 1))
        out_channels = in_channels * 2 // (2 if bilinear else 1)
        layers.append(Down(in_channels, out_channels, activation=activation, bias=bias))

        return nn.Sequential(*layers)

    def get_up_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        factor = 2 if bilinear else 1
        for i in range(depth - 1):
            in_channels = n_conv * (2**(depth - i))
            out_channels = in_channels // (2 * factor)
            layers.append(Up(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias))

        in_channels = n_conv * 2
        out_channels = n_conv
        layers.append(Up(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias))

        return nn.Sequential(*layers)

    def adapt_input_shape(self, x):
        in_shape = x.shape
        if len(in_shape) == 3:
            x = torch.unsqueeze(x, 0)
        return in_shape, x

    def adapt_output_shape(self, in_shape, x):
        if len(in_shape) == 3:
            assert x.shape[0] == 1
            x = x[0]
        return x

    def forward(self, x):
        in_shape, x = self.adapt_input_shape(x)

        x = self.inc(x)
        x_down = [x]
        for down in self.down:
            x = down(x)
            x_down.append(x)

        x = x_down.pop()

        for up in self.up:
            x = up(x, x_down.pop())

        x = self.outc(x)

        x = self.adapt_output_shape(in_shape, x)
        return x

class ViewSqueeze(nn.Module):
    def __init__(self):
        super(ViewUnsqueeze, self).__init__()

    def forward(self, x):
        return x.view(*x.shape[:-1])

class ViewUnsqueeze(nn.Module):
    def __init__(self):
        super(ViewUnsqueeze, self).__init__()

    def forward(self, x):
        return x.view(*x.shape, 1)

class UNet3d(UNet):
    def __init__(self, input_dimension, output_dimension,
                 n_conv=16, depth=4,
                 bilinear=False, activation=nn.LeakyReLU, bias=False, input_height=16, **kwargs):
        super(UNet3d, self).__init__(input_dimension, output_dimension,
                                     n_conv=n_conv, depth=depth,
                                     bilinear=bilinear, activation=activation, bias=bias, **kwargs)
        self.inc = DoubleConv3d(input_dimension, n_conv, activation=activation, bias=bias)
        self.outc = OutConv3d(n_conv, output_dimension, bias=bias)
        self.input_height = input_height

    def get_down_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        for i in range(depth - 1):
            in_channels = n_conv * (2**i)
            out_channels = in_channels * 2

            if self.input_height // (2**(i)) == 1:
                layers.append(ViewSqueeze())

            if self.input_height // (2**(i)) > 1:
                layer = Down3d(in_channels, out_channels, activation=activation, bias=bias)
            else:
                layer = Down(in_channels, out_channels, activation=activation, bias=bias)
            layers.append(layer)

        in_channels = n_conv * (2**(depth - 1))
        out_channels = in_channels * 2 // (2 if bilinear else 1)

        if self.input_height // (2**(depth - 1)) > 1:
            layer = Down3d(in_channels, out_channels, activation=activation, bias=bias)
        else:
            layer = Down(in_channels, out_channels, activation=activation, bias=bias)
        layers.append(layer)

        return nn.Sequential(*layers)

    def get_up_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        factor = 2 if bilinear else 1
        for i in range(depth - 1):
            in_channels = n_conv * (2**(depth - i))
            out_channels = in_channels // (2 * factor)

            if self.input_height // (2**(depth - i)) == 1:
                layers.append(ViewUnsqueeze())

            if self.input_height // (2**(depth - i)) > 1:
                layer = Up3d(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias)
            else:
                layer = Up(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias)
            layers.append(layer)

        in_channels = n_conv * 2
        out_channels = n_conv
        if self.input_height > 1:
            layer = Up3d(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias)
        else:
            layer = Up(in_channels, out_channels, bilinear=bilinear, activation=activation, bias=bias)
        layers.append(layer)

        return nn.Sequential(*layers)

    def adapt_input_shape(self, x):
        in_shape = x.shape
        if len(in_shape) == 4:
            x = torch.unsqueeze(x, 0)
        return in_shape, x

    def adapt_output_shape(self, in_shape, x):
        if len(in_shape) == 4:
            assert x.shape[0] == 1
            x = x[0]
        return x


class FCNCore(nn.Module):
    def __init__(self, fcn_core_layers=1, input_shape=[64, 64], n_conv=16, depth=4, extra_params_number=3, activation=nn.LeakyReLU, **kwargs):
        super(FCNCore, self).__init__()
        self.fcn_core = self.get_fcn_core(fcn_core_layers, input_shape, n_conv, depth, extra_params_number, activation)

    def get_fcn_core(self, fcn_core_layers, input_shape, n_conv, depth, extra_params_number, activation):
        if fcn_core_layers == 0:
            return None
        modules = []
        x_elements_number = 1
        for size in input_shape:
            x_elements_number *= size // (2**depth)
        for _ in range(fcn_core_layers):
            output_features = n_conv*(2**depth) * x_elements_number
            modules.append(nn.Linear(output_features + extra_params_number, output_features))
            modules.append(activation())
        return nn.Sequential(*modules)

    def call_fcn_core(self, x, extra_params):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        x = torch.cat((x, extra_params), dim=-1)
        x = self.fcn_core(x)
        x = x.reshape(shape)

        return x


class ConditionalUNet(UNet, FCNCore):
    def __init__(self, input_dimension, output_dimension,
                 input_shape=[64,64], n_conv=16, depth=4,
                 fcn_core_layers=1, extra_params_number: int=3,
                 bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(ConditionalUNet, self).__init__(input_dimension=input_dimension, output_dimension=output_dimension,
            input_shape=input_shape, fcn_core_layers=fcn_core_layers, extra_params_number=extra_params_number,
            n_conv=n_conv, depth=depth, bilinear=bilinear, activation=activation, bias=bias, **kwargs)
        print(fcn_core_layers, extra_params_number, self.fcn_core)


    def forward(self, x, extra_params):
        in_shape, x = self.adapt_input_shape(x)

        x = self.inc(x)
        x_down = [x]

        for down in self.down:
            x = down(x)
            x_down.append(x)

        x = x_down.pop()
        x = self.call_fcn_core(x, extra_params)

        for up in self.up:
            x = up(x, x_down.pop())

        x = self.outc(x)
        x = self.adapt_output_shape(in_shape, x)
        return x


class ConditionalUNet3d(UNet3d, FCNCore):
    def __init__(self, input_dimension, output_dimension,
                 input_shape=[64, 64, 16], n_conv=16, depth=4,
                 fcn_core_layers=1, extra_params_number: int=3,
                 bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(ConditionalUNet3d, self).__init__(input_dimension=input_dimension, output_dimension=output_dimension,
            input_shape=input_shape, fcn_core_layers=fcn_core_layers, extra_params_number=extra_params_number,
            n_conv=n_conv, depth=depth, bilinear=bilinear, activation=activation, bias=bias, **kwargs)

        print(fcn_core_layers, extra_params_number, self.fcn_core)


    def forward(self, x, extra_params):
        in_shape, x = self.adapt_input_shape(x)

        x = self.inc(x)
        x_down = [x]

        for down in self.down:
            x = down(x)
            x_down.append(x)

        x = x_down.pop()
        x = self.call_fcn_core(x, extra_params)

        for up in self.up:
            x = up(x, x_down.pop())

        x = self.outc(x)
        x = self.adapt_output_shape(in_shape, x)
        return x
