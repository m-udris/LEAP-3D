from torch import nn
import pytorch_lightning as pl

from leap3d.models import BaseModel, CNN


class LEAP3D_CNN(BaseModel):
    def __init__(self):
        super(LEAP3D_CNN, self).__init__()
        self.loss_function = nn.functional.mse_loss
        self.net = CNN()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_function(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
