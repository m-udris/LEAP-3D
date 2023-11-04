import pytorch_lightning as pl
from torch import optim

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

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self.f_step(batch, batch_idx, train=True)[0]

    def validation_step(self, batch, batch_idx):
        return self.f_step(batch, batch_idx, train=False)[1]

    def test_step(self, batch, batch_idx):
        return self.f_step(batch, batch_idx, train=False)[1]

    def log_loss(self, loss_name, loss, train, prog_bar=True):
        if train:
            self.log("train_" + loss_name, loss, prog_bar=prog_bar)
            self.log("train_epoch_" + loss_name, loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        else:
            self.log("val_" + loss_name, loss, prog_bar=prog_bar)
            self.log("val_epoch_" + loss_name, loss, prog_bar=prog_bar, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def f_step(self, batch, batch_idx, train):
        raise NotImplementedError("This is an abstract class.")
