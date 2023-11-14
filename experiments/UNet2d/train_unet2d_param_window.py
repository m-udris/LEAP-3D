import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from leap3d.dataset import ExtraParam, LEAP3DDataModule
from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.train import train_model


def train_unet2d_param_window(experiment_name='unet2d_param_window_5', lr=1e-3, fcn_core_layers=1, extra_params_number=3):
    hparams = {
        'batch_size': 256,
        'learning_rate': lr,
        'num_workers': 1,
        'max_epochs': 20,
        'architecture': Architecture.LEAP3D_UNET2D,
        'transforms': None,
        'in_channels': 3,
        'out_channels': 1,
        'window_size': 5,
        'window_step_size': 5,
        'fcn_core_layers': fcn_core_layers,
        'extra_params_number': extra_params_number,
        'activation': torch.nn.LeakyReLU
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
        DATA_DIR / "Params.npy", DATA_DIR / "Rough_coord.npz", DATA_DIR, DATASET_DIR,
        batch_size=hparams['batch_size'],
        train_cases=8, test_cases=[8,9],
        window_size=hparams['window_size'], window_step_size=hparams['window_step_size'],
        force_prepare=False,
        num_workers=hparams['num_workers'],
        extra_params=[ExtraParam.SCANNING_ANGLE, ExtraParam.LASER_POWER, ExtraParam.LASER_RADIUS],
        transform=None,
        extra_params_transform=None,
        target_transform=None,
    )

    model = hparams['architecture'].get_model(**hparams)
    checkpoint_filename = wandb_config['name']
    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, checkpoint_filename=checkpoint_filename, **hparams)

    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == '__main__':
    train_unet2d_param_window()
