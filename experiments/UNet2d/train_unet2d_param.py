import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from leap3d.dataset import LEAP3DDataModule, ExtraParam
from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.train import train_model, train


def train_unet2d_param(experiment_name='unet2d_with_params', *args, **kwargs):
    train(
        experiment_name=experiment_name,
        train_transforms=None,
        extra_params_transform=None,
        target_transforms=None,
        transform_target_to_train=None,
        *args,
        **kwargs
    )


if __name__ == '__main__':
    train_unet2d_param()
