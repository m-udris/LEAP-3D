import logging
from pathlib import Path
import sys

import torch.nn as nn

from leap3d.callbacks import get_checkpoint_only_last_epoch_callback, LogR2ScoreOverTimePlotCallback
from leap3d.dataset import ExtraParam, LEAP3DDataModule
from leap3d.models import Architecture
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform


def train_unet2d_param_window_normalized(experiment_name='unet2d_p_w10_l1+mse_loss', window_size=10, window_step_size=10, max_epochs=50, *args, **kwargs):
    loss_function = lambda y_hat, y: 0.5 * nn.functional.l1_loss(y_hat, y) + 0.5 * nn.functional.mse_loss(y_hat, y)
    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        loss_function=loss_function,
        *args,
        **kwargs
    )


if __name__ == '__main__':
    train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
