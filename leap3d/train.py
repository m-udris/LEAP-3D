import logging
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb
from leap3d.callbacks import LogR2ScoreOverTimePlotCallback, get_checkpoint_only_last_epoch_callback

from leap3d.dataset import ExtraParam, LEAP3DDataModule
from leap3d.models import Architecture, LEAP3D_UNet2D
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, MAX_LASER_POWER, MAX_LASER_RADIUS, MELTING_POINT, BASE_TEMPERATURE, NUM_WORKERS
from leap3d.models.unet2d import UNet2D
from leap3d.transforms import normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform, get_target_to_train_transform

DEFAULT_EXTRA_PARAMS = [ExtraParam.SCANNING_ANGLE, ExtraParam.LASER_POWER, ExtraParam.LASER_RADIUS]

IS_NAIVE_NORMALIZATION = False
TEMPERATURE_MIN = BASE_TEMPERATURE if not IS_NAIVE_NORMALIZATION else 0

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=TEMPERATURE_MIN, inplace=True))
    ])

DEFAULT_EXTRA_PARAMS_TRANSFORM = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: scanning_angle_cos_transform(x, 0, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 1, 0, MAX_LASER_POWER, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 2, 0, MAX_LASER_RADIUS, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 10, inplace=True))
    ])

DEFAULT_TARGET_TO_TRAIN_TRANSFORM = transforms.Compose([
        transforms.Lambda(lambda x: get_target_to_train_transform(TEMPERATURE_MIN, MELTING_POINT, 0, 10)(x, inplace=True))
    ])


def train(
        experiment_name: str,
        batch_size: int = 256,
        lr: float = 1e-3,
        num_workers: int = 1,
        max_epochs: int = 5,
        architecture: Architecture = Architecture.LEAP3D_UNET2D,
        train_transforms: transforms.Compose = DEFAULT_TRAIN_TRANSFORM,
        extra_params_transform: transforms.Compose = DEFAULT_EXTRA_PARAMS_TRANSFORM,
        target_transforms: transforms.Compose = DEFAULT_TARGET_TRANSFORM,
        transform_target_to_train: transforms.Compose = DEFAULT_TARGET_TO_TRAIN_TRANSFORM,
        in_channels: int = 3,
        out_channels: int = 1,
        window_size: int = 1,
        window_step_size: int = 1,
        fcn_core_layers: int = 1,
        extra_params: List[ExtraParam] = DEFAULT_EXTRA_PARAMS,
        target_max_temperature: int = 10,
        activation: torch.nn.Module = torch.nn.LeakyReLU,
        wandb_tags: List[str] = ['test'],
        eval_steps: int = 10,
        eval_samples: int = 100,
        force_prepare: bool = False,
        *args,
        **kwargs
):
    hparams = {
        'batch_size': batch_size,
        'lr': lr,
        'num_workers': num_workers,
        'max_epochs': max_epochs,
        'architecture': architecture,
        'transforms': 'default',
        'in_channels': in_channels,
        'out_channels': out_channels,
        'window_size': window_size,
        'window_step_size': window_step_size,
        'fcn_core_layers': fcn_core_layers,
        'extra_params_number': 0 if extra_params is None else len(extra_params),
        'extra_params': extra_params,
        'activation': activation,
        'target_max_temperature': target_max_temperature,
        'eval_steps': eval_steps,
        'eval_samples': eval_samples,
        'tags': wandb_tags,
        'force_prepare': force_prepare
    }

    # start a new wandb run to track this script
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': 'leap2d',
        # name of the run on wandb
        'name': experiment_name,
        # track hyperparameters and run metadata
        'config': hparams
    }

    logging.basicConfig(level=logging.DEBUG)
    wandb_logger = WandbLogger(log_model="all", **wandb_config)

    datamodule = LEAP3DDataModule(
        PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, DATASET_DIR,
        batch_size=hparams['batch_size'],
        train_cases=17, test_cases=[18,19], eval_cases=[20],
        window_size=hparams['window_size'], window_step_size=hparams['window_step_size'],
        num_workers=NUM_WORKERS,
        extra_params=extra_params,
        transform = train_transforms,
        target_transform = target_transforms,
        extra_params_transform = extra_params_transform,
        force_prepare=hparams['force_prepare']
    )

    model = hparams['architecture'].get_model(transform=transform_target_to_train, **hparams)
    checkpoint_filename = wandb_config['name']
    callbacks = [
        get_checkpoint_only_last_epoch_callback(checkpoint_filename),
        LogR2ScoreOverTimePlotCallback(steps=eval_steps, samples=eval_samples)]

    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, callbacks=callbacks, **hparams)

    trainer.test(model, datamodule=datamodule)

    wandb.finish()


def train_model(model: pl.LightningModule, datamodule: pl.LightningDataModule,
                max_epochs=10, logger: Logger=None, callbacks: List[pl.Callback]=None, **kwargs
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
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=callbacks, log_every_n_steps=25)
    trainer.fit(model, datamodule=datamodule)
    return trainer
