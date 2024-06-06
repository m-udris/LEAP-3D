import logging
from pathlib import Path
import sys

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import TEMPERATURE_MIN, train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform

import torch
from torchvision.transforms import transforms


DEFAULT_TRAIN_TRANSFORM_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=3000, base_temperature=TEMPERATURE_MIN, inplace=True))
    ])

DEFAULT_TRAIN_TRANSFORM_INVERSE_2D = transforms.Compose([
        # torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=3000, base_temperature=TEMPERATURE_MIN, inverse=True, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_2D = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 10, inplace=True))
    ])

DEFAULT_TARGET_TRANSFORM_INVERSE_2D = transforms.Compose([
        # torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, 10, inverse=True, inplace=True))
    ])

DEFAULT_TARGET_TO_TRAIN_TRANSFORM = transforms.Compose([
        transforms.Lambda(lambda x: get_target_to_train_transform(TEMPERATURE_MIN, 3000, 0, 10)(x, inplace=True))
    ])


def train_unet2d_param_window_normalized(experiment_name='unet2d_parallel_2', window_size=5, window_step_size=5, max_epochs=50, *args, **kwargs):
    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        loss_function='l1',
        batch_size=32,
        train_cases=list(range(0, 100, 2)),
        test_cases=list(range(1, 100, 10)),
        eval_cases=[5, 35, 65, 95],
        train_transforms=DEFAULT_TRAIN_TRANSFORM_2D,
        train_transforms_inverse=DEFAULT_TRAIN_TRANSFORM_INVERSE_2D,
        target_transforms=DEFAULT_TARGET_TRANSFORM_2D,
        target_transforms_inverse=DEFAULT_TARGET_TRANSFORM_INVERSE_2D,
        transform_target_to_train=DEFAULT_TARGET_TO_TRAIN_TRANSFORM,
        *args,
        **kwargs
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_unet2d_param_window_normalized()
    else:
        train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
