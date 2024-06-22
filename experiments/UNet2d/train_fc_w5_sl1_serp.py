import logging
from pathlib import Path
import sys

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform


def train_unet2d_param_window_normalized(experiment_name='unet2d_serpentine_w5-ws1_tanh', window_size=5, window_step_size=1, max_epochs=50, *args, **kwargs):
    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        loss_function='smooth_l1',
        batch_size=32,
        train_cases=list(range(0, 100, 2)),
        test_cases=list(range(1, 100, 10)),
        eval_cases=[5, 35, 65, 95],
        *args,
        **kwargs
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_unet2d_param_window_normalized()
    else:
        train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
