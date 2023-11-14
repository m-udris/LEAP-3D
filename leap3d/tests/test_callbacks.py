import logging
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import torch
from leap3d.callbacks import LogR2ScoreOverTimePlotCallback
from leap3d.config import DATA_DIR, DATASET_DIR, MELTING_POINT
from leap3d.dataset import LEAP3DDataModule
from leap3d.models import Architecture

def test_log_r2_score_over_time_plot_callback():
    hparams = {
        'batch_size': 256,
        'learning_rate': 1e-3,
        'num_workers': 1,
        'max_epochs': 5,
        'architecture': Architecture.LEAP3D_UNET2D,
        'transforms': None,
        'in_channels': 3,
        'out_channels': 1,
        'activation': torch.nn.LeakyReLU,
        'tags': ['test']
    }

    datamodule = LEAP3DDataModule(
        DATA_DIR / "Params.npy", DATA_DIR / "Rough_coord.npz", DATA_DIR, DATASET_DIR,
        batch_size=hparams['batch_size'],
        train_cases=8, test_cases=[8,9], eval_cases=[10],
        window_size=1, window_step_size=1,
        force_prepare=False,
        num_workers=hparams['num_workers'],
        transform=None,
        extra_params_transform=None,
        target_transform=None,
    )
    datamodule.prepare_data()
    datamodule.setup('test')

    model = hparams['architecture'].get_model(**hparams)

    callback = LogR2ScoreOverTimePlotCallback(steps=10)
    data = callback.get_r2_score_over_time(model, datamodule.leap_eval)

    assert all(map(lambda x: x[1] != 0, data))
