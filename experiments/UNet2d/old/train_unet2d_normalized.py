import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from leap3d.callbacks import LogR2ScoreOverTimePlotCallback, get_checkpoint_only_last_epoch_callback
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE
from leap3d.dataset import LEAP3DDataModule
from leap3d.models import Architecture
from leap3d.train import train_model
from leap3d.transforms import get_target_to_train_transform, normalize_temperature_2d


def train_unet2d_normalized(experiment_name='unet2d_normalized', lr=1e-3, target_max_temperature=10, max_epochs=5):
    hparams = {
        'batch_size': 256,
        'learning_rate': lr,
        'num_workers': 1,
        'max_epochs': max_epochs,
        'architecture': Architecture.LEAP3D_UNET2D,
        'transforms': None,
        'in_channels': 3,
        'out_channels': 1,
        'activation': torch.nn.LeakyReLU,
        'target_max_temperature': target_max_temperature,
        'tags': ['test']
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

    normalize_train_temperature = lambda x: normalize_temperature_2d(x, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True)
    normalize_target_temperature = lambda x: normalize_temperature_2d(x, target_max_temperature, base_temperature=0, inplace=True)

    train_transforms = transforms.Compose([
        torch.tensor,
        transforms.Lambda(normalize_train_temperature)
        # add more transforms as needed
    ])

    target_transforms = transforms.Compose([
        torch.tensor,
        transforms.Lambda(normalize_target_temperature)
        # add more transforms as needed
    ])

    target_to_train_transform = get_target_to_train_transform(BASE_TEMPERATURE, MELTING_POINT, 0, target_max_temperature)

    logging.basicConfig(level=logging.DEBUG)


    wandb_logger = WandbLogger(log_model="all", **wandb_config)
    datamodule = LEAP3DDataModule(
        DATA_DIR / "Params.npy", DATA_DIR / "Rough_coord.npz", DATA_DIR, DATASET_DIR,
        batch_size=hparams['batch_size'],
        train_cases=8, test_cases=[8,9], eval_cases=[10],
        window_size=1, window_step_size=1,
        force_prepare=False,
        num_workers=hparams['num_workers'],
        transform = train_transforms,
        target_transform = target_transforms
    )

    model = hparams['architecture'].get_model(transform=target_to_train_transform, **hparams)
    checkpoint_filename = wandb_config['name']
    callbacks = [
        get_checkpoint_only_last_epoch_callback(checkpoint_filename),
        LogR2ScoreOverTimePlotCallback(steps=10)]

    trainer = train_model(model=model, datamodule=datamodule, logger=wandb_logger, callbacks=callbacks, **hparams)

    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == '__main__':
    train_unet2d_normalized()
