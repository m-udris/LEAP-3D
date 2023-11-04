import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from leap3d.dataset import LEAP3DDataModule
from leap3d.models import Architecture, LEAP3D_UNet2D
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.models.unet2d import UNet2D


def train_model(model: pl.LightningModule, datamodule: pl.LightningDataModule,
                max_epochs=10, logger: Logger=None, checkpoint_filename: str=None, **kwargs
                ) -> pl.Trainer:
    """Train a model with a given architecture, learning rate, and number of epochs
    Args:
        model (pl.LightningModule): Lightning model to train
        datamodule (pl.LightningDataModule): Lightning datamodule to use for training
        max_epochs (int, optional): Number of epochs to train for. Defaults to 10.
        logger (pl.Logger, optional): Lightning logger to use for training. Defaults to None.
    Returns:
        trainer (pl.Trainer): Trainer used to train the model
    """
    callbacks = None
    if checkpoint_filename is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="./model_checkpoints/",
            filename=checkpoint_filename, monitor="step", mode="max",
            save_top_k=1
        )
        callbacks = [checkpoint_callback]
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    return trainer
