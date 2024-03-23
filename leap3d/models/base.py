from typing import Callable
import torch
from torch import nn, optim
from torchmetrics.regression import R2Score, MeanAbsoluteError
import pytorch_lightning as pl

from leap3d.losses import heat_loss

class BaseModel(pl.LightningModule):
    """ Base LEAP3D model.

    Kind of abstract class for the model used in the leap package.

    We define the training and validation steps. This is useful for uniformity.

    Parameters
    ----------
    net: nn.Module
        Neural network

    """
    def __init__(self, net=None, loss_function: str | Callable ='mse', *args, **kwargs) -> None:
        super().__init__()
        self.net = net
        self.save_hyperparameters(ignore=['net'])
        self.r2_metric = R2Score()
        self.mae_metric = MeanAbsoluteError()

        if loss_function == 'mse':
            self.loss_function = nn.functional.mse_loss
        elif loss_function == 'heat_loss':
            self.loss_function = heat_loss
        elif loss_function == 'l1':
            self.loss_function = nn.functional.l1_loss
        elif loss_function == 'smooth_l1':
            self.loss_function = nn.functional.smooth_l1_loss
        elif callable(loss_function):
            self.loss_function = loss_function
        else:
            raise ValueError(f"Unknown loss function {loss_function}")

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.f_step(batch, batch_idx, train=True, *args, **kwargs)[0]

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return self.f_step(batch, batch_idx, train=False, *args, **kwargs)[1]

    def test_step(self, batch, batch_idx, *args, **kwargs):
        return self.f_step(batch, batch_idx, train=False, *args, **kwargs)[1]

    def log_loss(self, loss_name, loss, train, prog_bar=True):
        prefix = "train_" if train else "val_"
        self.log(prefix + loss_name, loss, prog_bar=prog_bar)

    def log_metrics_dict(self, loss_dict, train, prog_bar=True):
        prefix = "train_" if train else "val_"
        loss_dict = {prefix + str(key): val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=prog_bar)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def f_step(self, batch, batch_idx, train):
        raise NotImplementedError("This is an abstract class.")
