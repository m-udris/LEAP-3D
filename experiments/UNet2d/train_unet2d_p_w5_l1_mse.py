import logging
from pathlib import Path
import sys

import torch.nn as nn

from leap3d.train import train


def train_unet2d_param_window_normalized(experiment_name='unet2d_p_w5_l1+mse_loss_b32', window_size=5, window_step_size=5, max_epochs=50, *args, **kwargs):
    loss_function = lambda y_hat, y: 0.5 * nn.functional.l1_loss(y_hat, y) + 0.5 * nn.functional.mse_loss(y_hat, y)
    train(
        experiment_name=experiment_name,
        window_size=window_size,
        window_step_size=window_step_size,
        max_epochs=max_epochs,
        loss_function=loss_function,
        batch_size=32,
        *args,
        **kwargs
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_unet2d_param_window_normalized()
    else:
        train_unet2d_param_window_normalized(dataset_dir=Path(sys.argv[1]))
