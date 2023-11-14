import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from leap3d.callbacks import get_checkpoint_only_last_epoch_callback, LogR2ScoreOverTimePlotCallback
from leap3d.dataset import LEAP3DDataModule
from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.train import train_model, train


def train_unet2d(experiment_name='unet2d_no_params', *args, **kwargs):
    train(
        experiment_name=experiment_name,
        train_transforms=None,
        extra_params_transform=None,
        target_transforms=None,
        transform_target_to_train=None,
        extra_params=None,
        **kwargs,
    )


if __name__ == '__main__':
    train_unet2d()
