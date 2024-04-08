import torch
import torch.nn as nn

from leap3d.models.unet import DoubleConv, DoubleConv3d, Down, Down3d


class CNN(nn.Module):
    """The downsampling part of a U-Net model."""
    def __init__(self, input_dimension, output_dimension, n_conv=16, depth=4, bilinear=False, activation=nn.LeakyReLU, bias=False, **kwargs):
        super(CNN, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_conv = n_conv
        self.bilinear = bilinear
        self.depth = depth

        self.inc = DoubleConv(input_dimension, n_conv, activation=activation, bias=bias)
        self.down = self.get_down_layers(n_conv, depth, activation=activation, bias=bias, bilinear=bilinear)

    def get_down_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        for i in range(depth - 1):
            in_channels = n_conv * (2**i)
            out_channels = in_channels * 2
            layers.append(self.get_down_layer(in_channels, out_channels, activation=activation, bias=bias))

        in_channels = n_conv * (2**(depth - 1))
        out_channels = in_channels * 2 // (2 if bilinear else 1)
        layer = Down(in_channels, out_channels, activation=activation, bias=bias)
        layers.append(layer)

        return nn.Sequential(*layers)

    def get_down_layer(self, in_channels, out_channels, activation, bias=False):
        """Only used for backward compatibility for older checkpoints."""
        return Down(in_channels, out_channels, activation=activation, bias=bias)

    def get_output_features_count(self, input_shape):
        x_elements_number = 1
        for size in input_shape:
            x_elements_number *= max(1, size // (2**self.depth))
        output_features = self.n_conv*(2**self.depth) * x_elements_number
        return output_features

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
        x = self.down(x)

        return self.adapt_output_shape(in_shape, x)


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(*x.shape[:-1])

class CNN3D(CNN):
    """The downsampling part of a U-Net model."""
    def __init__(self, input_dimension, output_dimension, n_conv=16, depth=4, bilinear=False, activation=nn.LeakyReLU, bias=False, input_height=16, **kwargs):
        self.input_height = input_height
        super(CNN3D, self).__init__(input_dimension, output_dimension, n_conv=n_conv, depth=depth, bilinear=bilinear, activation=activation, bias=bias, **kwargs)
        self.inc = DoubleConv3d(input_dimension, n_conv, activation=activation, bias=bias)

    def get_down_layers(self, n_conv, depth, activation, bias=False, bilinear=False):
        layers = []
        for i in range(depth - 1):
            in_channels = n_conv * (2**i)
            out_channels = in_channels * 2

            if self.input_height // (2**(i)) == 1:
                layers.append(View())

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
