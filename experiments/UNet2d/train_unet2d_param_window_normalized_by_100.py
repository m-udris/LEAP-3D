import logging
from pathlib import Path
import sys

import torch
from torchvision import transforms

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform


def train_unet2d_param_window_normalized(experiment_name='unet2d_p_w5_nt100_heat_loss', window_size=5, window_step_size=5, max_epochs=50, *args, **kwargs):
    target_transform = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 100, inplace=True))
    ])

    target_to_train_transform = transforms.Compose([
        transforms.Lambda(lambda x: get_target_to_train_transform(BASE_TEMPERATURE, MELTING_POINT, 0, 100)(x, inplace=True))
    ])

    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        target_transforms=target_transform,
        target_to_train_transforms=target_to_train_transform,
        loss_function='heat_loss',
        *args,
        **kwargs
    )


if __name__ == '__main__':
    train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
