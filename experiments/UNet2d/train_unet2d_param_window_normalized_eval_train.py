import logging
from pathlib import Path
import sys

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform


def train_unet2d_param_window_normalized(experiment_name='unet2d_param_window_5_normalized_eval_train', window_size=5, window_step_size=5, max_epochs=100, *args, **kwargs):
    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        eval_cases=[0],
        *args,
        **kwargs
    )


if __name__ == '__main__':
    train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
