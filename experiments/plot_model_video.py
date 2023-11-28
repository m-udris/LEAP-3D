from pathlib import Path
import sys

from matplotlib import animation

from leap3d.scanning import ScanParameters, ScanResults
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, NUM_WORKERS
from leap3d.plotting import plot_model_top_layer_temperature_comparison
from leap3d.dataset import LEAP3DDataModule
from leap3d.models import LEAP3D_UNet2D
from leap3d.train import DEFAULT_EXTRA_PARAMS, DEFAULT_TRAIN_TRANSFORM_2D, DEFAULT_TARGET_TRANSFORM_2D, DEFAULT_EXTRA_PARAMS_TRANSFORM, DEFAULT_TARGET_TRANSFORM_INVERSE_2D, DEFAULT_TRAIN_TRANSFORM_INVERSE_2D

case_params = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=20)

plot_dir = Path("./plots/")
dataset_dir = Path(sys.argv[1])

checkpoint_filepath = Path(sys.argv[2])
model = LEAP3D_UNet2D.load_from_checkpoint(checkpoint_filepath)
model.eval()

train_cases = 18
test_cases = [18, 19]
eval_cases = [20]

datamodule = LEAP3DDataModule(
        PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, dataset_dir,
        is_3d=False,
        batch_size=128,
        train_cases=train_cases, test_cases=test_cases, eval_cases=eval_cases,
        window_size=10, window_step_size=10,
        num_workers=NUM_WORKERS,
        extra_params=DEFAULT_EXTRA_PARAMS,
        transform = DEFAULT_TRAIN_TRANSFORM_2D,
        target_transform = DEFAULT_TARGET_TRANSFORM_2D,
        extra_params_transform = DEFAULT_EXTRA_PARAMS_TRANSFORM,
        transform_inverse = DEFAULT_TRAIN_TRANSFORM_INVERSE_2D,
        target_transform_inverse = DEFAULT_TARGET_TRANSFORM_INVERSE_2D,
        force_prepare=False
    )
datamodule.prepare_data()
datamodule.setup(stage="")

dataset = datamodule.leap_eval

fig, axes, ims = plot_model_top_layer_temperature_comparison(case_params, model, dataset, steps=len(dataset) - 1, samples=1)

ani = animation.ArtistAnimation(fig, ims, repeat=True, interval=20, blit=True, repeat_delay=1000)

animation_filepath = plot_dir / f"predicted_top_layer_temperature_for_whole_case.gif"
ani.save(animation_filepath, fps=10, progress_callback=lambda frame_number, total_frames: print(f"{frame_number}/{total_frames}", end="\r"))
