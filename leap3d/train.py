import logging
from pathlib import Path
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb
from leap3d.callbacks import LogR2ScoreOverTimePlotCallback, PlotErrorOverTimeCallback, PlotTopLayerTemperatureCallback, Rollout2DUNetCallback, get_checkpoint_only_last_epoch_callback

# from leap3d.dataset import Channel, ExtraParam, LEAP3DDataModule
from leap3d.datasets.channels import RoughTemperature, LaserPosition, RoughTemperatureDifference, ScanningAngle, LaserPower, LaserRadius, Channel
from leap3d.datasets.unet_forecasting import UNetForecastingDataModule

from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, MAX_LASER_POWER, MAX_LASER_RADIUS, MELTING_POINT, BASE_TEMPERATURE, NUM_WORKERS, FORCE_PREPARE
from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.transforms import normalize_extra_param, normalize_temperature_2d, normalize_temperature_3d, scanning_angle_cos_transform, get_target_to_train_transform

DEFAULT_EXTRA_PARAMS = [ScanningAngle, LaserPower, LaserRadius]

IS_NAIVE_NORMALIZATION = False
TEMPERATURE_MIN = BASE_TEMPERATURE if not IS_NAIVE_NORMALIZATION else 0

DEFAULT_TRAIN_TRANSFORM_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=TEMPERATURE_MIN, inplace=True))
    ])

DEFAULT_TRAIN_TRANSFORM_INVERSE_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=TEMPERATURE_MIN, inverse=True, inplace=True))
    ])

DEFAULT_TRAIN_TRANSFORM_3D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_3d(x, melting_point=MELTING_POINT, base_temperature=TEMPERATURE_MIN, inplace=True))
    ])

DEFAULT_TRAIN_TRANSFORM_INVERSE_3D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_3d(x, melting_point=MELTING_POINT, base_temperature=TEMPERATURE_MIN, inverse=True, inplace=True))
    ])

DEFAULT_EXTRA_PARAMS_TRANSFORM = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: scanning_angle_cos_transform(x, 0, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 1, 0, MAX_LASER_POWER, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 2, 0, MAX_LASER_RADIUS, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 10, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_INVERSE_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 10, inverse=True, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_3D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_3d(x, 10, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_INVERSE_3D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_3d(x, 10, inverse=True, inplace=True))
    ])

DEFAULT_TARGET_TO_TRAIN_TRANSFORM = transforms.Compose([
        transforms.Lambda(lambda x: get_target_to_train_transform(TEMPERATURE_MIN, MELTING_POINT, 0, 10)(x, inplace=True))
    ])


def train(
        experiment_name: str,
        project_name: str = 'leap2d',
        batch_size: int = 64,
        lr: float = 1e-3,
        num_workers: int = 1,
        max_epochs: int = 5,
        architecture: Architecture = 'default',
        train_transforms: transforms.Compose = 'default',
        extra_params_transform: transforms.Compose = DEFAULT_EXTRA_PARAMS_TRANSFORM,
        target_transforms: transforms.Compose = 'default',
        train_transforms_inverse: transforms.Compose = 'default',
        target_transforms_inverse: transforms.Compose = 'default',
        transform_target_to_train: transforms.Compose = DEFAULT_TARGET_TO_TRAIN_TRANSFORM,
        input_channels: List[Channel] = [LaserPosition, RoughTemperature],
        target_channels: List[Channel] = [RoughTemperatureDifference],
        in_channels: int = 3,
        out_channels: int = 1,
        window_size: int = 1,
        window_step_size: int = 1,
        fcn_core_layers: int = 1,
        extra_input_channels: List[Channel] = DEFAULT_EXTRA_PARAMS,
        target_max_temperature: int = 10,
        activation: torch.nn.Module = torch.nn.LeakyReLU,
        wandb_tags: List[str] = ['test'],
        eval_steps: int = 10,
        eval_samples: int = 100,
        force_prepare: bool = FORCE_PREPARE,
        is_3d: bool = False,
        train_cases: int | List[int] = 18,
        test_cases: int | List[int] = [18, 19],
        eval_cases: int | List[int] = [20],
        no_callbacks: bool = False,
        loss_function: str = 'mse',
        *args,
        **kwargs
):
    if train_transforms == 'default':
        train_transforms = DEFAULT_TRAIN_TRANSFORM_3D if is_3d else DEFAULT_TRAIN_TRANSFORM_2D
    if target_transforms == 'default':
        target_transforms = DEFAULT_TARGET_TRANSFORM_3D if is_3d else DEFAULT_TARGET_TRANSFORM_2D
    if architecture == 'default':
        architecture = Architecture.LEAP3D_UNET3D if is_3d else Architecture.LEAP3D_UNET2D
    if train_transforms_inverse == 'default':
        train_transforms_inverse = DEFAULT_TRAIN_TRANSFORM_INVERSE_3D if is_3d else DEFAULT_TRAIN_TRANSFORM_INVERSE_2D
    if target_transforms_inverse == 'default':
        target_transforms_inverse = DEFAULT_TARGET_TRANSFORM_INVERSE_3D if is_3d else DEFAULT_TARGET_TRANSFORM_INVERSE_2D

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
        'extra_params_number': 0 if extra_input_channels is None else len(extra_input_channels),
        'extra_input_channels': extra_input_channels,
        'activation': activation,
        'target_max_temperature': target_max_temperature,
        'eval_steps': eval_steps,
        'eval_samples': eval_samples,
        'tags': wandb_tags,
        'force_prepare': force_prepare,
        'is_3d': is_3d,
        'padding_mode': 'replicate',
        'loss_function': loss_function,
        'input_height': kwargs.get('input_height', 16),
    }

    # start a new wandb run to track this script
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': project_name,
        # name of the run on wandb
        'name': experiment_name,
        # track hyperparameters and run metadata
        'config': hparams
    }

    logging.basicConfig(level=logging.INFO)
    wandb_logger = WandbLogger(log_model="all", **wandb_config)

    transforms = {
        'input': train_transforms,
        'target': target_transforms,
        'extra_input': extra_params_transform,
    }
    inverse_transforms = {
        'input': train_transforms_inverse,
        'target': target_transforms_inverse,
    }

    datamodule = UNetForecastingDataModule(
        PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, kwargs.get('dataset_dir', DATASET_DIR / 'unet_forecasting'),
        is_3d=is_3d,
        batch_size=hparams['batch_size'],
        train_cases=train_cases, test_cases=test_cases, eval_cases=eval_cases,
        window_size=hparams['window_size'], window_step_size=hparams['window_step_size'],
        num_workers=NUM_WORKERS,
        extra_input_channels=extra_input_channels,
        transforms=transforms,
        inverse_transforms=inverse_transforms,
        force_prepare=hparams['force_prepare'],
        input_channels=input_channels,
        target_channels=target_channels,
    )

    plot_dir = Path("./plots/")
    scan_parameters = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, 20)

    model = hparams['architecture'].get_model(transform=transform_target_to_train, **hparams)
    checkpoint_filename = wandb_config['name']
    callbacks = [
        get_checkpoint_only_last_epoch_callback(checkpoint_filename)
    ]
    if not no_callbacks:
        callbacks += [
            # LogR2ScoreOverTimePlotCallback(steps=eval_steps, samples=eval_samples),

            # Rollout2DUNetCallback(scan_parameters)
        ]

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
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=callbacks, log_every_n_steps=25, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=datamodule)
    return trainer
