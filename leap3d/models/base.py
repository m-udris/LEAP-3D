import pytorch_lightning as pl
from torch import optim
from torchmetrics.regression import R2Score, MeanAbsoluteError


class BaseModel(pl.LightningModule):
    """ Base LEAP3D model.

    Kind of abstract class for the model used in the leap package.

    We define the training and validation steps. This is useful for uniformity.

    Parameters
    ----------
    net: nn.Module
        Neural network

    """
    def __init__(self, net=None) -> None:
        super().__init__()
        self.net = net
        self.save_hyperparameters(ignore=['net'])
        self.r2_metric = R2Score()
        self.mae_metric = MeanAbsoluteError()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return self(batch)

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
