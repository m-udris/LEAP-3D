import logging
from pathlib import Path
import sys

import torch
from torchvision.transforms import transforms

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_log_to_train_transform, get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, normalize_temperature_log_2d, scanning_angle_cos_transform


TEMPERATURE_MIN = 300
TEMPERATURE_MAX = 3000
TEMPERATURE_DIFF_MIN = -750
TEMPERATURE_DIFF_MAX = 1500

input_transform = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=TEMPERATURE_MAX, base_temperature=TEMPERATURE_MIN, inplace=True))
    ])

input_transform_inverse = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=TEMPERATURE_MAX, base_temperature=TEMPERATURE_MIN, inverse=True, inplace=True))
    ])

target_transform = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_log_2d(x, TEMPERATURE_DIFF_MIN, max_value=TEMPERATURE_DIFF_MAX, inplace=True))
    ])

target_transform_inverse = transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_log_2d(x, TEMPERATURE_DIFF_MIN, max_value=TEMPERATURE_DIFF_MAX, inverse=True, inplace=True))
    ])

target_to_train = transforms.Compose([
        transforms.Lambda(lambda x: get_target_log_to_train_transform(TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DIFF_MIN, TEMPERATURE_DIFF_MAX)(x, inplace=True))
    ])


def train_unet2d_param_window_normalized(experiment_name='fc_w5ws5_input_target-log_norm', window_size=5, window_step_size=5, max_epochs=50, *args, **kwargs):
    with torch.autograd.detect_anomaly():
        train(
            experiment_name=experiment_name,
            window_size=window_size,
            window_step_size=window_step_size,
            max_epochs=max_epochs,
            loss_function='l1',
            batch_size=32,
            train_cases=list(range(0, 100, 2)),
            test_cases=list(range(1, 100, 10)),
            # eval_cases=[5, 35, 65, 95],
            eval_cases=[],
            train_transforms=input_transform,
            target_transforms=target_transform,
            train_transforms_inverse=input_transform_inverse,
            target_transforms_inverse=target_transform_inverse,
            transform_target_to_train=target_to_train,
            *args,
            **kwargs
        )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_unet2d_param_window_normalized()
    else:
        train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))


# python ./experiments/UNet2d/train_fc_w5_l1_serp_i-n-t-log.py