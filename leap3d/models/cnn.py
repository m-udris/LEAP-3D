import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dimension, outsize, out_channels, channels, H_init=4, W_init=2, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU()):
        super(CNN, self).__init__()
        self.input_dimension = input_dimension
        self.channels = channels
        self.Hinit = H_init
        self.Winit = W_init
        out_FCN_dim = channels[0]*self.Hinit*self.Winit
        self.out_FCN_dim = out_FCN_dim
        self.out_channels = out_channels
        self.upsample_scale = 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.outsize = outsize

        # FCN to send input to latent space
        self.FCN = nn.Sequential(nn.Linear(self.input_dimension, out_FCN_dim),
                                 self.activation,
                                 nn.Linear(out_FCN_dim, out_FCN_dim),
                                 self.activation,
                                 nn.Linear(out_FCN_dim, out_FCN_dim))
        self.conv = nn.ModuleList()

        for i in range(len(channels)-1):
            self.conv.append(nn.Sequential(nn.Upsample(scale_factor=self.upsample_scale, mode='bilinear'),
                                            nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=(self.kernel_size, self.kernel_size), padding=(self.padding, self.padding), groups=1, padding_mode='replicate'),
                                            self.activation))

        self.output_layer = nn.Sequential(nn.Upsample(size=self.outsize, mode='bilinear'),
                                          nn.Conv2d(in_channels=channels[-1], out_channels=self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), padding=(self.padding, self.padding), groups=1, padding_mode='replicate'),
                                          self.activation)

    def forward(self, x):
        # performs convolution and non-linear transformations defining the network
        in_shape = x.shape
        x = x.view(-1, self.input_dimension)
        x = self.FCN(x)
        x = x.view(-1, self.channels[0], self.Hinit, self.Winit)

        for module in self.conv:
            x = module(x)

        x = self.output_layer(x)
        if len(in_shape) == 1:
            assert x.shape[0] == 1
            x = x[0]
        x = put_channels_last(x)
        return x
