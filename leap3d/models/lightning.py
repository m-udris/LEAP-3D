import logging
from typing import Callable

import numpy as np
import torch
from torch import nn

from leap3d.models import BaseModel, CNN, MLP, UNet, UNet3d, ConditionalUNet, ConditionalUNet3d
from leap3d.losses import distance_l1_loss_2d, heat_loss, smooth_l1_loss, weighted_l1_loss, heavy_weighted_l1_loss
from leap3d.positional_encoding import positional_encoding


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
    def __init__(self, net, lr=1e-3, loss_function: str | Callable ='mse', apply_positional_encoding=False, positional_encoding_L=8, *args, **kwargs):
        if loss_function == 'wl1':
            loss_function = weighted_l1_loss
        if loss_function == 'hwl1':
            loss_function = heavy_weighted_l1_loss
        super(InterpolationUNet, self).__init__(net, loss_function=loss_function, *args, **kwargs)
        self.apply_positional_encoding = apply_positional_encoding
        self.positional_encoding_L = positional_encoding_L
        self.lr = lr

    def forward(self, x, extra_params):
        return self.net(x, extra_params)

    def f_step(self, batch, batch_idx, train=False, *args, **kwargs):
        x, extra_params, y = batch[:3]

        if self.apply_positional_encoding:
            if x.shape[1] > 1:
                new_x_coords = positional_encoding(x[:,:-1], L=self.positional_encoding_L).reshape((x.shape[0], 2 * self.positional_encoding_L * (x.shape[1] - 1), *x.shape[2:]))
                x = torch.cat((new_x_coords, x[:,-1].unsqueeze(1)), dim=1)

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
    def __init__(self, input_shape=[32,32], extra_params_number=3,
                 input_dimension=1, output_dimension=1,
                 n_conv=16, depth=4,
                 hidden_layers=[1024, 2048, 2048],
                 apply_positional_encoding=False, positional_encoding_L=3,
                 activation=nn.LeakyReLU, bias=False, *args, **kwargs):
        super(InterpolationMLP, self).__init__(*args, **kwargs)
        self.cnn = CNN(input_dimension=input_dimension, output_dimension=output_dimension, n_conv=n_conv, depth=depth, activation=activation, bias=bias, **kwargs)

        self.apply_positional_encoding = apply_positional_encoding
        self.positional_encoding_L = positional_encoding_L

        mlp_input_size = self.cnn.get_output_features_count(input_shape) + extra_params_number
        mlp_input_size += 2 * (2 * self.positional_encoding_L if self.apply_positional_encoding else 1)

        self.mlp = MLP(input_dimension=mlp_input_size, output_dimension=output_dimension, hidden_layers=hidden_layers, activation=activation, bias=bias, **kwargs)
        self.input_shape = input_shape

    def forward(self, x_grid, points):
        x = self.forward_cnn(x_grid)
        x = self.forward_mlp(x, points)
        return x

    def forward_cnn(self, x_grid):
        x_grid = self.cnn(x_grid)
        x = x_grid.reshape(x_grid.shape[0], -1)
        return x

    def forward_mlp(self, x_conv, points):
        # x = x.unsqueeze(1).repeat(1, points.shape[1], 1)
        x_conv = x_conv.unsqueeze(1)
        x = x_conv.expand(-1, points.shape[1], *x_conv.shape[2:])

        x = torch.cat((x, points), dim=-1)
        return self.mlp(x)

    def f_step(self, batch, batch_idx, train=False, *args, **kwargs):
        if len(batch) > 5:
            return self.f_step_with_melt_pool(batch, batch_idx, train=train, *args, **kwargs)
        x, extra_params, y, target_coordinates, laser_data = batch[:5]

        point_coords = target_coordinates

        if self.apply_positional_encoding:
            point_coords = positional_encoding(point_coords, L=self.positional_encoding_L).reshape(*point_coords.shape[:2], -1)

        # extra_params = extra_params.unsqueeze(1).repeat(1, point_coords.shape[1], 1)
        extra_params = extra_params.unsqueeze(1)
        extra_params_expanded = extra_params.expand((-1, point_coords.shape[1], *extra_params.shape[2:]))

        points = torch.cat((point_coords, extra_params_expanded), dim=2)

        x = x.to(self.device)
        points = points.to(self.device)

        y_hat = self.forward(x, points)
        y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y = y.reshape(y_hat.shape)

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
        x, extra_params, y, target_coordinates, laser_data, melting_pool = batch[:6]

        point_coords = target_coordinates

        if self.apply_positional_encoding:
            point_coords = positional_encoding(point_coords, L=self.positional_encoding_L).reshape(*point_coords.shape[:2], -1)

        # === Compute in parallel, then filter out for loss === (Slower)

        # y = y.reshape(y.shape[0], -1)
        # melting_pool_coords = melting_pool[:, :, :2]
        # melting_pool_temps = melting_pool[:, :, 3]

        # point_coords = torch.cat((point_coords, melting_pool_coords), dim=1)
        # print(y.shape, melting_pool_temps.shape)
        # y = torch.cat((y, melting_pool_temps), dim=1)

        # extra_params = extra_params.unsqueeze(1)
        # extra_params_expanded = extra_params.expand((-1, point_coords.shape[1], *extra_params.shape[2:]))

        # points = torch.cat((point_coords, extra_params_expanded), dim=2)

        # x = x.to(self.device)
        # points = points.to(self.device)

        # y_hat = self.forward(x, points)

        # y_hat = y_hat.reshape(y_hat.shape[0], -1)
        # # y = y.reshape(y_hat.shape)

        # mask = torch.logical_and(torch.logical_and(points[:, :, 0] > 0, points[:, :, 1] > 0), y > 0).reshape(y.shape)

        # y_hat = y_hat[mask]
        # y = y[mask]

        # === Filter and compute in sequence, then concatenate === (Faster)

        x_grids = self.forward_cnn(x)

        temperature_list = []
        y_hat_list = []
        for x_grid, extra_params_item, point_coords_item, y_points, melt_points in zip(x_grids, extra_params, point_coords, y, melting_pool):
            melt_point_filter_sum = torch.abs(melt_points[:, 0]) + torch.abs(melt_points[:, 1]) + torch.abs(melt_points[:, 3])
            melt_point_filter = torch.where(melt_point_filter_sum > 0)
            melt_points = melt_points[melt_point_filter]

            melt_point_coords = melt_points[:, :2]
            if self.apply_positional_encoding:
                melt_point_coords = positional_encoding(melt_point_coords, L=self.positional_encoding_L).reshape(*melt_point_coords.shape[:-1], -1)
            new_coord_entry = torch.cat((point_coords_item.reshape(-1, point_coords_item.shape[-1]), melt_point_coords), dim=0)
            new_coord_entry = new_coord_entry.to(self.device)

            new_coord_entry = new_coord_entry

            # extra_params_item = extra_params_item.repeat(new_coord_entry.shape[0], 1)
            extra_params_item_expanded = extra_params_item.expand((new_coord_entry.shape[0], *extra_params_item.shape))

            points = torch.cat((new_coord_entry, extra_params_item_expanded), dim=-1)

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

        # === End of filtering and concatenation ===

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


class InterpolationMLPChunks(BaseModel):
    def __init__(self, input_shape=[32,32], extra_params_number=3,
                 in_channels=1, out_channels=1,
                 n_conv=16, depth=4,
                 hidden_layers=[1024],
                 apply_positional_encoding=False, positional_encoding_L=3,
                 activation=nn.LeakyReLU, bias=False,
                 return_gradients=False, learn_gradients=False, multiply_gradients_by=1, predict_cooldown_rate=False,
                 temperature_loss_weight=1, pos_grad_loss_weight=1, temporal_grad_loss_weight=1,
                 *args, **kwargs):

        print(kwargs.get('in_channels'))
        super(InterpolationMLPChunks, self).__init__(*args, **kwargs)

        self.cnn = CNN(input_dimension=in_channels, output_dimension=out_channels, n_conv=n_conv, depth=depth, activation=activation, bias=bias, **kwargs)

        self.apply_positional_encoding = apply_positional_encoding
        self.positional_encoding_L = positional_encoding_L

        self.extra_params_number = extra_params_number

        mlp_input_size = self.cnn.get_output_features_count(input_shape) + extra_params_number
        mlp_input_size += 2 * (2 * self.positional_encoding_L if self.apply_positional_encoding else 1)

        self.mlp = MLP(input_dimension=mlp_input_size, output_dimension=out_channels, hidden_layers=hidden_layers, activation=activation, bias=bias, **kwargs)
        self.input_shape = input_shape

        self.return_gradients = return_gradients
        self.learn_gradients = learn_gradients
        self.multiply_gradients_by = multiply_gradients_by

        self.predict_cooldown_rate = predict_cooldown_rate

        self.temperature_loss_weight = temperature_loss_weight
        self.pos_grad_loss_weight = pos_grad_loss_weight
        self.temporal_grad_loss_weight = temporal_grad_loss_weight

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x_grid, points):
        input_points = points
        input_x_grid = x_grid
        if self.return_gradients:
            input_points = input_points.requires_grad_()
            input_x_grid = input_x_grid.requires_grad_()

        if self.apply_positional_encoding:
            x_grid, points = self.positional_encoding(x_grid, points)

        x_cnn = self.forward_cnn(x_grid)
        x = self.forward_mlp(x_cnn, points)

        if self.return_gradients:
            x_sum = torch.sum(x[..., 0])
            grads = torch.autograd.grad(x_sum, points, create_graph=True)[0]
            grads = grads * self.multiply_gradients_by
            return x, grads

        return x

    def positional_encoding(self, x_grid, points):
        if x_grid.shape[1] > 1:
            new_x_coords = positional_encoding(x_grid[:,:-1], L=self.positional_encoding_L).reshape((x_grid.shape[0], 2 * self.positional_encoding_L * (x_grid.shape[1] - 1), *x_grid.shape[2:]))
            x_grid = torch.cat((new_x_coords, x_grid[:,-1].unsqueeze(1)), dim=1)
        encoded_coordinates = positional_encoding(points[:, :, :-self.extra_params_number], L=self.positional_encoding_L).reshape(*points[:, :, :self.extra_params_number].shape[:2], -1)
        points = torch.cat((encoded_coordinates, points[:, :, -self.extra_params_number:]), dim=2)

        return x_grid, points

    def compute_gradients(self, x, points):
        x_sum = torch.sum(x)
        grads = torch.autograd.grad(x_sum, points, create_graph=True)

        return grads[0]

    def forward_cnn(self, x_grid):
        x_grid = self.cnn(x_grid)
        x = x_grid.reshape(x_grid.shape[0], -1)
        return x

    def forward_mlp(self, x_conv, points):
        x_conv = x_conv.unsqueeze(1)
        x = x_conv.expand(-1, points.shape[1], *x_conv.shape[2:])

        x = torch.cat((x, points), dim=-1)
        return self.mlp(x)

    def f_step(self, batch, batch_idx, train=False, *args, **kwargs):
        x, extra_params, target = batch

        point_coordinates = target[:, :, :2]
        y = target[:, :, 2]

        if self.learn_gradients:
            true_grads_x = target[:, :, 3]
            true_grads_y = target[:, :, 4]

        if self.predict_cooldown_rate:
            true_grads_t = target[:, :, -1]

        extra_params = extra_params.unsqueeze(1)
        extra_params_expanded = extra_params.expand((-1, point_coordinates.shape[1], *extra_params.shape[2:]))

        points = torch.cat((point_coordinates, extra_params_expanded), dim=2)

        if self.return_gradients:
            model_output, grads = self.forward(x, points)
        else:
            model_output = self.forward(x, points)

        y_hat = model_output[..., 0]

        y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y = y.reshape(y_hat.shape)

        main_loss = self.loss_function(y_hat, y)

        loss = self.temperature_loss_weight * main_loss

        if self.learn_gradients:
            mask = torch.isnan(true_grads_x)
            grads[:,:,0] /= self.multiply_gradients_by
            grads[:,:,1] /= self.multiply_gradients_by
            true_grads_x /= self.multiply_gradients_by
            true_grads_y /= self.multiply_gradients_by
            grad_x_loss = smooth_l1_loss(grads[:, :, 0][~mask], true_grads_x[~mask])
            grad_y_loss = smooth_l1_loss(grads[:, :, 1][~mask], true_grads_y[~mask])
            grad_x_r2 = self.r2_metric(grads[:, :, 0][~mask], true_grads_x[~mask])
            grad_y_r2 = self.r2_metric(grads[:, :, 1][~mask], true_grads_y[~mask])
            loss += self.pos_grad_loss_weight * (grad_x_loss + grad_y_loss)

        if self.predict_cooldown_rate:
            mask = torch.isnan(true_grads_t)
            true_grads_t = true_grads_t[~mask]
            predicted_grads_t = model_output[..., 1][~mask]
            grad_t_loss = self.loss_function(predicted_grads_t, true_grads_t)
            loss += self.temporal_grad_loss_weight * grad_t_loss
            grad_t_r2 = self.r2_metric(predicted_grads_t, true_grads_t)

        metrics_dict = {
            "loss": loss,
            "r2": self.r2_metric(y_hat.reshape(-1), y.reshape(-1)),
            "mae": self.mae_metric(y_hat, y),
            "heat_loss": heat_loss(y_hat, y),
            "mse": nn.functional.mse_loss(y_hat, y),
        }
        if self.learn_gradients:
            metrics_dict["main_loss"] = main_loss
            metrics_dict["grad_x_loss"] = grad_x_loss
            metrics_dict["grad_y_loss"] = grad_y_loss
            metrics_dict["grad_x_r2"] = grad_x_r2
            metrics_dict["grad_y_r2"] = grad_y_r2

        if self.predict_cooldown_rate:
            metrics_dict["grad_t_loss"] = grad_t_loss
            metrics_dict["grad_t_r2"] = grad_t_r2

        self.log_metrics_dict(metrics_dict, train)

        if self.return_gradients:
            return loss, y_hat, grads, true_grads_x, true_grads_y

        return loss, y_hat
