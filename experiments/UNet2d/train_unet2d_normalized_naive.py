import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from leap3d.dataset import LEAP3DDataModule
from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.train import train_model
from leap3d.transforms import normalize_temperature_2d


def train_unet2d_normalized_naive():
    hparams = {
        'batch_size': 256,
        'learning_rate': 1e-3,
        'num_workers': 1,
        'max_epochs': 20,
        'architecture': Architecture.LEAP3D_UNET2D,
        'transforms': None,
        'in_channels': 3,
        'out_channels': 1,
    }

    # start a new wandb run to track this script
    wandb_config = {
        # set the wandb project where this run will be logged
        'project': 'leap2d',
        # name of the run on wandb
        'name': 'unet2d_normalized_naive_fixed',
        # track hyperparameters and run metadata
        'config': hparams
    }

    normalize_temperature = lambda x: normalize_temperature_2d(x, MELTING_POINT, base_temperature=0, inplace=True)

    train_transforms = transforms.Compose([
        torch.tensor,
        transforms.Lambda(normalize_temperature)
    ])

    target_transforms = transforms.Compose([
        torch.tensor,
        transforms.Lambda(normalize_temperature)
    ])

    logging.basicConfig(level=logging.DEBUG)


    wandb_logger = WandbLogger(log_model="all", **wandb_config)
    datamodule = LEAP3DDataModule(
        DATA_DIR / "Params.npy", DATA_DIR / "Rough_coord.npz", DATA_DIR, DATASET_DIR,
        batch_size=hparams['batch_size'],
        train_cases=8, test_cases=[8,9],
        window_size=1, window_step_size=1,
        force_prepare=False,
        num_workers=hparams['num_workers'],
        transform = train_transforms,
        target_transform = target_transforms
    )

    model = hparams['architecture'].get_model(**hparams)
    checkpoint_filename = wandb_config['name']
    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, checkpoint_filename=checkpoint_filename, **hparams)

    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == '__main__':
    train_unet2d_normalized_naive()
