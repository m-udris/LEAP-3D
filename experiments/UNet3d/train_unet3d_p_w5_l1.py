import logging
from pathlib import Path
import sys

from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT, BASE_TEMPERATURE, MAX_LASER_POWER, MAX_LASER_RADIUS
from leap3d.train import train_model, train
from leap3d.transforms import get_target_to_train_transform, normalize_extra_param, normalize_temperature_2d, scanning_angle_cos_transform


def train_unet3d_param_window_normalized(experiment_name='unet3d_p_w5_l1_loss', window_size=5, window_step_size=5, max_epochs=50, *args, **kwargs):
    train(
        experiment_name=experiment_name,
        project_name='leap3d',
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        loss_function='l1',
        is_3d=True,
        batch_size=16,
        train_cases=list(range(0, 200, 2)),
        test_cases=list(range(1, 200, 10)),
        eval_cases=[5, 35, 65, 95, 105, 135, 165, 195],
        *args,
        **kwargs
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_unet3d_param_window_normalized()
    else:
        train_unet3d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
