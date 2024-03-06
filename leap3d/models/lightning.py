import logging
from typing import Callable

import numpy as np
import torch
from torch import nn

from leap3d.models import BaseModel, CNN, MLP, UNet, UNet3d, ConditionalUNet, ConditionalUNet3d
from leap3d.losses import distance_l1_loss_2d, heat_loss, weighted_l1_loss, heavy_weighted_l1_loss



class LEAP3D_UNet(BaseModel):
    def __init__(self, net, lr=1e-3, transform=None, loss_function: str | Callable ='mse', is_teacher_forcing: bool=False, *args, **kwargs):
        super(LEAP3D_UNet, self).__init__(net=net, loss_function=loss_function, *args, **kwargs)
        self.transform = transform

        self.is_teacher_forcing = is_teacher_forcing
        self.lr = lr

    def forward(self, x, extra_params):
        return self.net(x, extra_params)

    def f_step(self, batch, batch_idx, train=False, steps=None, *args, **kwargs):
        x_window, extra_params_window, y_window = batch
        if len(x_window.shape) == 4:
            return self.f_step_single(x_window, extra_params_window, y_window, train=train)
        if x_window.shape[1] == 1:
            return self.f_step_single(x_window[:, 0], extra_params_window[:, 0], y_window[:, 0], train=train)
        return self.f_step_window(x_window, extra_params_window, y_window, train=train)

    def f_step_single(self, x, extra_params, y, train=False, log_loss=True):
        y_hat = self.net(x, extra_params)
        y = y.reshape(y_hat.shape)
        loss = self.loss_function(y_hat, y)

        if log_loss:
            metrics_dict = {
                "loss": loss,
                "r2": self.r2_metric(y_hat.reshape(-1), y.reshape(-1)),
                "mae": self.mae_metric(y_hat, y),
                # "heat_loss": heat_loss(y_hat, y),
                "mse": nn.functional.mse_loss(y_hat, y),
            }
            self.log_metrics_dict(metrics_dict, train)

        return loss, y_hat

    def f_step_window(self, x_window, extra_params_window, y_window, train=False):
        loss = 0
        predicted_x_temperature = None
        outputs = []

        for i in range(x_window.shape[1]):
            x = x_window[:, i]
            extra_params = extra_params_window[:, i]
            y = y_window[:, i]
            if not self.is_teacher_forcing and predicted_x_temperature is not None:
                x = x.clone()
                x[:, -1] = predicted_x_temperature
            loss_single, y_hat = self.f_step_single(x, extra_params, y, log_loss=False)
            outputs.append(y_hat)
            loss += loss_single
            predicted_x_temperature = self.get_predicted_temperature(x[:, -1], y_hat[:, 0])

        y_hat_window = torch.stack(outputs, dim=1)
        metrics_dict = {
            "loss": loss,
            "r2": self.r2_metric(y_hat_window.reshape(-1), y_window.reshape(-1)),
            "mae": self.mae_metric(y_hat_window, y_window),
            # "heat_loss": heat_loss(y_hat, y),
            "mse": nn.functional.mse_loss(y_hat, y),
        }
        self.log_metrics_dict(metrics_dict, train)

        return loss, outputs

    def get_predicted_temperature(self, x, y_hat):
        if self.transform is not None:
            return x + self.transform(y_hat)
        return x + y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class LEAP3D_UNet2D(LEAP3D_UNet):
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, transform=None, is_teacher_forcing=False, *args, **kwargs):
        if "fcn_core_layers" not in kwargs:
            net = UNet(in_channels, out_channels, **kwargs)
        else:
            net = ConditionalUNet(in_channels, out_channels, **kwargs)

        super(LEAP3D_UNet2D, self).__init__(net, lr=lr, transform=transform, is_teacher_forcing=is_teacher_forcing, *args, **kwargs)

class LEAP3D_UNet3D(LEAP3D_UNet):
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, transform=None, *args, **kwargs):
        if "fcn_core_layers" not in kwargs:
            net = UNet3d(in_channels, out_channels, **kwargs)
        else:
            net = ConditionalUNet3d(in_channels, out_channels, **kwargs)
        super(LEAP3D_UNet3D, self).__init__(net, lr=lr, transform=transform, *args, **kwargs)
        self.is_teacher_forcing = False


class InterpolationUNet(BaseModel):
    def __init__(self, net, lr=1e-3, loss_function: str | Callable ='mse', *args, **kwargs):
        if loss_function == 'wl1':
            loss_function = weighted_l1_loss
        if loss_function == 'hwl1':
            loss_function = heavy_weighted_l1_loss
        super(InterpolationUNet, self).__init__(net, loss_function=loss_function, *args, **kwargs)

        self.lr = lr

    def forward(self, x, extra_params):
        return self.net(x, extra_params)

    def f_step(self, batch, batch_idx, train=False, *args, **kwargs):
        x, extra_params, y = batch[:3]
        y_hat = self.net(x, extra_params)
        y = y.reshape(y_hat.shape)
        if self.loss_function == distance_l1_loss_2d:
            loss_weights = batch[3]
            loss = self.loss_function(y_hat, y, loss_weights)
        else:
            loss = self.loss_function(y_hat, y)

        metrics_dict = {
            "loss": loss,
            "r2": self.r2_metric(y_hat.reshape(-1), y.reshape(-1)),
            "mae": self.mae_metric(y_hat, y),
            "heat_loss": heat_loss(y_hat, y),
            "mse": nn.functional.mse_loss(y_hat, y),
        }
        self.log_metrics_dict(metrics_dict, train)

        return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class InterpolationUNet2D(InterpolationUNet):
    def __init__(self, in_channels=1, out_channels=1, lr=1e-3, loss_function: str | Callable ='mse', *args, **kwargs):
        net = ConditionalUNet(in_channels, out_channels, **kwargs)
        if loss_function == 'dl1':
            loss_function = distance_l1_loss_2d
        super(InterpolationUNet2D, self).__init__(net, lr=lr, loss_function=loss_function, *args, **kwargs)

class InterpolationUNet3D(InterpolationUNet):
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, loss_function: str | Callable ='mse', *args, **kwargs):
        net = ConditionalUNet3d(in_channels, out_channels, **kwargs)
        super(InterpolationUNet2D, self).__init__(net, lr=lr, loss_function=loss_function, *args, **kwargs)


class InterpolationMLP(BaseModel):
    def __init__(self, input_shape=[32,32], extra_params_number=3, input_dimension=1, output_dimension=1, n_conv=16, depth=4, hidden_layers=[1024, 2048, 2048], activation=nn.LeakyReLU, bias=False, *args, **kwargs):
        super(InterpolationMLP, self).__init__(*args, **kwargs)
        self.cnn = CNN(input_dimension=input_dimension, output_dimension=output_dimension, n_conv=n_conv, depth=depth, activation=activation, bias=bias, **kwargs)
        mlp_input_size = self.cnn.get_output_features_count(input_shape)
        self.mlp = MLP(input_dimension=mlp_input_size + extra_params_number + 2, output_dimension=output_dimension, hidden_layers=hidden_layers, activation=activation, bias=bias, **kwargs)
        self.input_shape = input_shape

    def forward(self, x_grid, points):
        x = self.forward_cnn(x_grid)
        x = self.forward_mlp(x, points)
        return x

    def forward_cnn(self, x_grid):
        x_grid = self.cnn(x_grid)
        x = x_grid.reshape(x_grid.shape[0], -1)
        return x

    def forward_mlp(self, x, points):
        # x = x.unsqueeze(1).repeat(1, points.shape[1], 1)
        x = x.unsqueeze(1).expand(-1, points.shape[1], *x.shape[1:])

        x = torch.cat((x, points), dim=-1)
        return self.mlp(x)

    def f_step(self, batch, batch_idx, train=False, *args, **kwargs):
        if len(batch) > 5:
            return self.f_step_with_melt_pool(batch, batch_idx, train=train, *args, **kwargs)
        x, extra_params, y, y_coord_bounds, laser_data = batch[:5]
        target_coord_points = []
        for coord_point in y_coord_bounds:
            x_coords = torch.linspace(coord_point[0], coord_point[1], 128)
            y_coords = torch.linspace(coord_point[2], coord_point[3], 128)
            target_points = torch.tensor(np.array(np.meshgrid(x_coords, y_coords))).mT.reshape(-1, 2)
            target_coord_points.append(target_points.unsqueeze(0))
        point_coords = torch.cat(target_coord_points, dim=0).to(self.device)


        # extra_params = extra_params.unsqueeze(1).repeat(1, point_coords.shape[1], 1)
        extra_params = extra_params.unsqueeze(1).expand((-1, point_coords.shape[1], *extra_params.shape[1:]))

        points = torch.cat((point_coords, extra_params), dim=2)

        x = x.to(self.device)
        points = points.to(self.device)

        y_hat = self.forward(x, points)
        y_hat = y_hat.reshape(y_hat.shape[0], -1)

        loss = self.loss_function(y_hat, y)

        metrics_dict = {
            "loss": loss,
            "r2": self.r2_metric(y_hat.reshape(-1), y.reshape(-1)),
            "mae": self.mae_metric(y_hat, y),
            "heat_loss": heat_loss(y_hat, y),
            "mse": nn.functional.mse_loss(y_hat, y),
        }
        self.log_metrics_dict(metrics_dict, train)

        return loss, y_hat

    def f_step_with_melt_pool(self, batch, batch_idx, train=False, *args, **kwargs):
        x, extra_params, y, y_coord_bounds, laser_data, melting_pool = batch[:6]
        target_coord_points = []
        for coord_point in y_coord_bounds:
            x_coords = torch.linspace(coord_point[0], coord_point[1], self.input_shape[0] * 4)
            y_coords = torch.linspace(coord_point[2], coord_point[3], self.input_shape[1] * 4)
            target_points = torch.tensor(np.array(np.meshgrid(x_coords, y_coords))).mT.reshape(-1, 2)
            target_coord_points.append(target_points.unsqueeze(0))
        point_coords = torch.cat(target_coord_points, dim=0).to(self.device)

        x_grids = self.forward_cnn(x)

        temperature_list = []
        y_hat_list = []
        for x_grid, extra_params_item, point_coords_item, y_points, melt_points, laser_data_points in zip(x_grids, extra_params, point_coords, y, melting_pool, laser_data):
            melt_point_filter_sum = torch.abs(melt_points[:, 0]) + torch.abs(melt_points[:, 1]) + torch.abs(melt_points[:, 3])
            melt_point_filter = torch.where(melt_point_filter_sum > 0)
            melt_points = melt_points[melt_point_filter]

            melt_point_coords = melt_points[:, :2]
            melt_point_coords -= laser_data_points[:2]
            new_coord_entry = torch.cat((point_coords_item.reshape(-1, 2), melt_point_coords), dim=0)
            new_coord_entry = new_coord_entry.to(self.device)

            new_coord_entry = new_coord_entry

            # extra_params_item = extra_params_item.repeat(new_coord_entry.shape[0], 1)
            extra_params_item = extra_params_item.expand((new_coord_entry.shape[0], *extra_params_item.shape))

            points = torch.cat((new_coord_entry, extra_params_item), dim=-1)

            points = points.unsqueeze(0)
            x_grid = x_grid.unsqueeze(0)
            y_hat = self.forward_mlp(x_grid, points)
            y_hat_list.append(y_hat.flatten())

            melt_point_temps = melt_points[:, 3]
            new_temp_entry = torch.cat((y_points.flatten(), melt_point_temps))
            new_temp_entry = new_temp_entry.to(self.device)
            temperature_list.append(new_temp_entry)

        y_hat = torch.stack(y_hat_list, dim=0).flatten()
        y = torch.stack(temperature_list, dim=0).flatten()

        loss = self.loss_function(y_hat, y)

        metrics_dict = {
            "loss": loss,
            "r2": self.r2_metric(y_hat.reshape(-1), y.reshape(-1)),
            "mae": self.mae_metric(y_hat, y),
            "heat_loss": heat_loss(y_hat, y),
            "mse": nn.functional.mse_loss(y_hat, y),
        }
        self.log_metrics_dict(metrics_dict, train)

        return loss, y_hat










