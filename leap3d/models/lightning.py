import logging

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, MeanAbsoluteError

from leap3d.models import BaseModel, CNN, UNet2D, UNet3D


class LEAP3D_CNN(BaseModel):
    def __init__(self):
        super(LEAP3D_CNN, self).__init__()
        self.loss_function = nn.functional.mse_loss
        self.net = CNN()
        self.r2_metric = R2Score()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_function(y_hat, y)

        self.r2_metric(y_hat.squeeze(), y.squeeze())

        self.log_dict({"train_loss": loss, "train_r2": self.r2_metric})

        return loss


# TODO: LOSS
#  (p_diff - gt_diff)**2 * (|gt_diff| + eps)
# eps = max |gt_diff| / 20
def heat_loss(y_hat, y, eps=0.05):
    max_diff = torch.max(torch.max(torch.abs(y), dim=-1).values, dim=-1, keepdims=True).values
    new_eps = max_diff * eps
    return torch.mean((y_hat - y) ** 2 * (torch.abs(y) + new_eps))
class LEAP3D_UNet(BaseModel):
    def __init__(self, net, in_channels=4, out_channels=1, lr=1e-3, transform=None, loss_function='mse', *args, **kwargs):
        super(LEAP3D_UNet, self).__init__(net, in_channels, out_channels, *args, **kwargs)
        self.transform = kwargs.get("transform", transform)
        if loss_function == 'mse':
            self.loss_function = nn.functional.mse_loss
        if loss_function == 'heat_loss':
            self.loss_function = heat_loss
        self.is_teacher_forcing = False
        self.lr = lr

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
                "mae": self.mae_metric(y_hat, y)
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
            "mae": self.mae_metric(y_hat_window, y_window)
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
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, transform=None, *args, **kwargs):
        net = UNet2D(in_channels, out_channels, **kwargs)
        super(LEAP3D_UNet2D, self).__init__(net, in_channels, out_channels, lr=lr, transform=transform, *args, **kwargs)
        # self.transform = kwargs.get("transform", transform)
        # self.loss_function = nn.functional.mse_loss
        self.is_teacher_forcing = False
        # self.lr = lr

class LEAP3D_UNet3D(LEAP3D_UNet):
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, transform=None, *args, **kwargs):
        net = UNet3D(in_channels, out_channels, **kwargs)
        super(LEAP3D_UNet3D, self).__init__(net, in_channels, out_channels, lr=lr, transform=transform, *args, **kwargs)
        self.is_teacher_forcing = False
