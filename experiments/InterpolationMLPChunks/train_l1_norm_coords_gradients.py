import logging
from pathlib import Path
import sys
from typing import List

import torch
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import transforms
import wandb

from leap3d.callbacks import get_checkpoint_only_last_epoch_callback
from leap3d.datasets.channels import LowResRoughTemperatureAroundLaser, MeltPoolPointChunk, ScanningAngle, LaserPower, LaserRadius
from leap3d.datasets import MLPInterpolationChunkDataModule
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, MAX_LASER_POWER, MAX_LASER_RADIUS, MELTING_POINT, BASE_TEMPERATURE, NUM_WORKERS, FORCE_PREPARE, X_MIN, X_MAX
from leap3d.models.lightning import InterpolationMLPChunks
from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.train import train_model
from leap3d.transforms import normalize_extra_param, normalize_temperature_2d, normalize_temporal_grad, scanning_angle_cos_transform, normalize_positional_grad


def train():
    # step_size = (X_MAX - X_MIN) / 64
    scan_parameters = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, 0)
    step_size = scan_parameters.rough_coordinates_step_size
    coords_radius = step_size * 16

    TEMPERATURE_MAX = MELTING_POINT
    TEMPERATURE_MAX = 2950
    # TEMPERATURE_MAX = 2400
    LASER_RADIUS_MAX = coords_radius
    # GRAD_T_MAX = 80_000_000
    GRAD_T_MAX = 300_000_000
    # GRAD_T_MAX = (2950 - 300) * 100_000

    MULTIPLY_GRADIENTS_BY = (TEMPERATURE_MAX - BASE_TEMPERATURE) / step_size

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_DIR / 'mlp_interpolation_chunks_coordinates_gradients'

    activation = nn.Tanh

    hparams = {
        'batch_size': 128,
        'lr': 1e-3,
        'num_workers': NUM_WORKERS,
        'max_epochs': 100,
        'transforms': 'default',
        'in_channels': 1,
        'out_channels': 2,
        'extra_params_number': 3,
        'input_channels': [LowResRoughTemperatureAroundLaser(return_coordinates=False)],
        'target_channels': [MeltPoolPointChunk(is_3d=False, chunk_size=32*32, input_shape=[32,32])],
        'extra_params': [ScanningAngle, LaserPower, LaserRadius],
        'tags': ['MLP', '2D', 'interpolation', 'chunks', 'l1_loss', 'norm_coords', 'all_gradients', f'T_max_{TEMPERATURE_MAX}', f'grad_t_max_{GRAD_T_MAX}'],
        'force_prepare': False,
        'is_3d': False,
        'padding_mode': 'replicate',
        'loss_function': 'l1',
        'input_shape': [32, 32],
        'target_shape': [3],
        'hidden_layers': [256, 256, 256, 256],
        'apply_positional_encoding': True,
        'positional_encoding_L': 16,
        'return_gradients': True,
        'learn_gradients': True,
        'multiply_gradients_by': MULTIPLY_GRADIENTS_BY,
        'predict_cooldown_rate': True,
        'temperature_max': TEMPERATURE_MAX,
        'laser_radius_max': LASER_RADIUS_MAX,
        'grad_t_max': GRAD_T_MAX,
        'activation': activation,
        'temperature_loss_weight': 1,
        'pos_grad_loss_weight': 1,
        'temporal_grad_loss_weight': 3,
    }

    # start a new wandb run to track this script
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': 'leap2d',
        # name of the run on wandb
        'name': f'LeakyReLU_mlp_{hparams["loss_function"]}_{hparams["hidden_layers"]}_all_grads',
        # track hyperparameters and run metadata
        'config': hparams
    }

    logging.basicConfig(level=logging.INFO)
    logging.debug('Starting wandb logger')
    wandb_logger = WandbLogger(log_model="all", **wandb_config)

    train_transforms = {
        'input': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=TEMPERATURE_MAX, base_temperature=BASE_TEMPERATURE, inplace=True)),
            # transforms.Lambda(lambda x: normalize_temperature_2d(x, temperature_channel_index=0, melting_point=coords_radius, base_temperature=-coords_radius, inplace=True)),
            # transforms.Lambda(lambda x: normalize_temperature_2d(x, temperature_channel_index=1, melting_point=coords_radius, base_temperature=-coords_radius, inplace=True))
        ]),
        'target': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_extra_param(x, index=2, min_value=BASE_TEMPERATURE, max_value=TEMPERATURE_MAX, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, index=0, min_value=-coords_radius, max_value=coords_radius, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, index=1, min_value=-coords_radius, max_value=coords_radius, inplace=True)),
            transforms.Lambda(lambda x: normalize_positional_grad(x, index=3, norm_constant=MULTIPLY_GRADIENTS_BY, inplace=True)),
            transforms.Lambda(lambda x: normalize_positional_grad(x, index=4, norm_constant=MULTIPLY_GRADIENTS_BY, inplace=True)),


            # transforms.Lambda(lambda x: normalize_positional_grad(x, index=3, max_temp=MELTING_POINT, min_temp=BASE_TEMPERATURE, coord_radius=coords_radius, inplace=True)),
            # transforms.Lambda(lambda x: normalize_positional_grad(x, index=4, max_temp=MELTING_POINT, min_temp=BASE_TEMPERATURE, coord_radius=coords_radius, inplace=True)),
            transforms.Lambda(lambda x: normalize_temporal_grad(x, index=-1, max_temp=1, min_temp=0, norm_constant=GRAD_T_MAX, inplace=True)),

        ]),
        'extra_input': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: scanning_angle_cos_transform(x, 0, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, 1, 0, MAX_LASER_POWER, inplace=True)),
            transforms.Lambda(lambda x: normalize_extra_param(x, 2, 0, LASER_RADIUS_MAX, inplace=True))
        ])
    }

    inverse_transforms = {
        'input': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=TEMPERATURE_MAX, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True))
        ]),
        'target': transforms.Compose([
            torch.tensor,
            transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=TEMPERATURE_MAX, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True)),
        ])
    }

    logging.debug('Creating datamodule')
    datamodule = MLPInterpolationChunkDataModule(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, dataset_dir,
                    is_3d=False, batch_size=hparams['batch_size'],
                    train_cases=18, test_cases=[18, 19],
                    input_shape=[32, 32], target_shape=[3],
                    extra_input_channels=hparams['extra_params'], input_channels=hparams['input_channels'], target_channels=hparams['target_channels'],
                    transforms=train_transforms, inverse_transforms=inverse_transforms,
                    force_prepare=False, num_workers=NUM_WORKERS)

    logging.debug('Creating model')
    model = InterpolationMLPChunks(**hparams)
    checkpoint_filename = wandb_config['name']
    callbacks = [
        get_checkpoint_only_last_epoch_callback(checkpoint_filename, monitor='val_loss', mode='min')
    ]
    logging.debug('Training model')
    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, callbacks=callbacks, **hparams)

    # trainer.test(model, datamodule=datamodule)

    wandb.finish()

if __name__ == '__main__':
    train()
