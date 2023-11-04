import logging

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, MeanAbsoluteError

from leap3d.models import BaseModel, CNN, UNet2D


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


class LEAP3D_UNet2D(BaseModel):
    def __init__(self, in_channels=4, out_channels=1, lr=1e-3, **kwargs):
        net = UNet2D(in_channels, out_channels)
        super(LEAP3D_UNet2D, self).__init__(net)
        self.lr = lr
        self.loss_function = nn.functional.mse_loss
        self.r2_metric = R2Score()
        self.mae_metric = MeanAbsoluteError()

    def f_step(self, batch, batch_idx, train):
        x, y = batch
        y_hat = self.net(x)
        y = y.reshape(y_hat.shape)
        loss = self.loss_function(y_hat, y)
        self.r2_metric(y_hat.reshape(-1), y.reshape(-1))
        self.mae_metric(y_hat, y)

        if train:
            self.log_dict({"train_loss": loss, "train_r2": self.r2_metric, "train_mae": self.mae_metric})
        else:
            self.log_dict({"val_loss": loss, "val_r2": self.r2_metric, "val_mae": self.mae_metric})

        return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
