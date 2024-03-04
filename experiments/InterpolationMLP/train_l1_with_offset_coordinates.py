import logging
from pathlib import Path
import sys
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb
from leap3d.callbacks import LogR2ScoreOverTimePlotCallback, PlotErrorOverTimeCallback, PlotTopLayerTemperatureCallback, Rollout2DUNetCallback, get_checkpoint_only_last_epoch_callback

from leap3d.datasets.channels import OffsetLowResRoughTemperatureAroundLaser, OffsetTemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius
from leap3d.datasets import MLPInterpolationDataModule
from leap3d.models import Architecture, LEAP3D_UNet2D
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, MAX_LASER_POWER, MAX_LASER_RADIUS, MELTING_POINT, BASE_TEMPERATURE, NUM_WORKERS, FORCE_PREPARE
from leap3d.models.lightning import InterpolationMLP
from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.train import train_model
from leap3d.transforms import normalize_extra_param, normalize_temperature_2d, normalize_temperature_3d, scanning_angle_cos_transform, get_target_to_train_transform


def train():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_DIR / 'mlp_interpolation_offset_no_distances_no_coordinates'

    hparams = {
        'batch_size': 32,
        'lr': 1e-3,
        'num_workers': NUM_WORKERS,
        'max_epochs': 100,
        'transforms': 'default',
        'in_channels': 3,
        'input_dimension': 3,
        'out_channels': 1,
        'extra_params_number': 3,
        'input_channels': [OffsetLowResRoughTemperatureAroundLaser(return_coordinates=True)],
        'target_channels': [OffsetTemperatureAroundLaser],
        'extra_params': [ScanningAngle, LaserPower, LaserRadius],
        'activation': torch.nn.LeakyReLU,
        'tags': ['MLP', '2D', 'interpolation', 'l1_loss', 'with_coordinates', 'offset'],
        'force_prepare': False,
        'is_3d': False,
        'padding_mode': 'replicate',
        'loss_function': 'l1',
        'input_shape': [24, 24],
        'target_shape': [24*4, 24*4]
    }

    # start a new wandb run to track this script
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': 'leap2d',
        # name of the run on wandb
        'name': f'interpolation_mlp_2d_l1_loss_b{hparams["batch_size"]}_offset_with_coordinates',
        # track hyperparameters and run metadata
        'config': hparams
    }

    logging.basicConfig(level=logging.INFO)
    wandb_logger = WandbLogger(log_model="all", **wandb_config)

    train_transforms = {
        'input': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True))
        ]),
        'target': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True))
        ]),
        'extra_params': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: scanning_angle_cos_transform(x, 0, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, 1, 0, MAX_LASER_POWER, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, 2, 0, MAX_LASER_RADIUS, inplace=True))
        ]),
        'melting_pool': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_extra_param(x, 3, max_value=MELTING_POINT, min_value=BASE_TEMPERATURE, inplace=True))
        ])
    }

    inverse_transforms = {
        'input': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True))
        ]),
        'target': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True))
        ])
    }

    datamodule = MLPInterpolationDataModule(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, dataset_dir,
                    is_3d=False, batch_size=hparams['batch_size'],
                    train_cases=18, test_cases=[18, 19],
                    input_shape=hparams['input_shape'], target_shape=hparams['target_shape'],
                    extra_input_channels=hparams['extra_params'], input_channels=hparams['input_channels'], target_channels=hparams['target_channels'],
                    transforms=train_transforms, inverse_transforms=inverse_transforms,
                    include_melting_pool=True,
                    include_distances_to_melting_pool=False,
                    force_prepare=False, num_workers=NUM_WORKERS)

    # datamodule.prepare_data()
    # datamodule.setup()
    model = InterpolationMLP(**hparams)
    checkpoint_filename = wandb_config['name']
    callbacks = [
        get_checkpoint_only_last_epoch_callback(checkpoint_filename, monitor='val_loss', mode='min', filename='mlp_l1_offset_coords')
    ]
    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, callbacks=callbacks, **hparams)

    trainer.test(model, datamodule=datamodule)

    wandb.finish()

if __name__ == '__main__':
    train()
