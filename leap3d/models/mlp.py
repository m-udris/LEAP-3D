import torch.nn as nn

class MLP(nn.Module):
    """A simple FCN model."""
    def __init__(self, input_dimension, output_dimension=1, hidden_layers=[1024], activation=nn.LeakyReLU, bias=False, **kwargs):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers

        self.fcn = self.get_layers(input_dimension, output_dimension, hidden_layers, activation=activation, bias=bias)

    def get_layers(self, input_dimension, output_dimension, hidden_layers, activation, bias=False):
        layers = []
        input_size = input_dimension
        for layer in hidden_layers:
            output_size = layer
            layers.append(nn.Linear(input_size, output_size, bias=bias))
            layers.append(activation())
            input_size = output_size

        layers.append(nn.Linear(input_size, output_dimension, bias=bias))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fcn(x)

        return x
