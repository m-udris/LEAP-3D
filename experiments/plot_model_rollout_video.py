from pathlib import Path
import sys

from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from leap3d.scanning import ScanParameters
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, NUM_WORKERS
from leap3d.plotting import plot_top_layer_temperature
from leap3d.dataset import LEAP3DDataModule
from leap3d.models import LEAP3D_UNet2D
from leap3d.callbacks import get_recursive_model_predictions, get_r2_scores
from leap3d.train import DEFAULT_EXTRA_PARAMS, DEFAULT_TRAIN_TRANSFORM_2D, DEFAULT_TARGET_TRANSFORM_2D, DEFAULT_EXTRA_PARAMS_TRANSFORM, DEFAULT_TARGET_TRANSFORM_INVERSE_2D, DEFAULT_TRAIN_TRANSFORM_INVERSE_2D, DEFAULT_TARGET_TO_TRAIN_TRANSFORM

scan_params = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=20)
plot_dir = Path("./plots/")

model_name = "unet2d_p_w5_l1_loss_b32"
model_name = sys.argv[1]

checkpoint_filepath = f"./model_checkpoints/{model_name}.ckpt"

model = LEAP3D_UNet2D.load_from_checkpoint(checkpoint_filepath)
model.transform = DEFAULT_TARGET_TO_TRAIN_TRANSFORM
model.eval()

datamodule = LEAP3DDataModule(
        PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, DATASET_DIR,
        is_3d=False,
        batch_size=128,
        train_cases=[], test_cases=[], eval_cases=[20],
        window_size=5, window_step_size=5,
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

x_gt_values = []
y_values = []
y_hat_values = []
x_pred_values = []
for x, y, y_hat, x_pred_temperature in get_recursive_model_predictions(model, dataset):
    x_gt_values.append(x)
    y_values.append(y)
    y_hat_values.append(y_hat)
    x_pred_values.append(x_pred_temperature)

r2_temperature_scores, r2_temperature_numerators, r2_temperature_denominators = get_r2_scores(x_pred_values, x_gt_values[1:], return_parts=True)
r2_temperature_difference_scores, r2_temperature_difference_numerators, r2_temperature_difference_denominators =  get_r2_scores(y_hat_values, y_values, return_parts=True)

timestep = 0

fig, axes = plt.subplots(ncols=2, nrows=4)
fig.set_size_inches(15, 15)
for ax in axes.flatten():
    ax.set_aspect('equal')
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

def plot_model_at_timestep(timestep, setup=False):
    x = x_gt_values[timestep]
    y = y_values[timestep]
    next_x_gt = x_gt_values[timestep + 1]
    next_x_gt_calculated = model.get_predicted_temperature(x, y)

    y_hat = y_hat_values[timestep]
    next_x_p = x_pred_values[timestep]

    r2_temperature = r2_temperature_scores[timestep]
    r2_temperature_numerator = r2_temperature_numerators[timestep]
    r2_temperature_denominator = r2_temperature_denominators[timestep]

    r2_temperature_difference = r2_temperature_difference_scores[timestep]
    r2_temperature_difference_numerator = r2_temperature_difference_numerators[timestep]
    r2_temperature_difference_denominator = r2_temperature_difference_denominators[timestep]

    fig.suptitle(f"timestep={timestep}")

    ax1.set_title("Ground Truth Temperature")
    im1 = plot_top_layer_temperature(ax1, x, scan_params, vmin=0, vmax=1)

    ax2.set_title("Ground Truth Next Temperature")
    im2 = plot_top_layer_temperature(ax2, next_x_gt, scan_params, vmin=0, vmax=1)

    ax3.set_title("Calculated Ground Truth Next Temperature")
    im3 = plot_top_layer_temperature(ax3, next_x_gt_calculated, scan_params, vmin=0, vmax=1)

    ax4.set_title("Ground Truth Temperature Difference")
    im4 = plot_top_layer_temperature(ax4, y, scan_params, vmin=-100, vmax=100, cmap='seismic')

    ax5.set_title(f"Predicted Next Temperature, R2={r2_temperature:.3f} ({r2_temperature_numerator:.6f}/{r2_temperature_denominator:.6f})")
    im5 = plot_top_layer_temperature(ax5, next_x_p, scan_params, vmin=0, vmax=1)

    ax6.set_title(f"Predicted Temperature Difference, R2={r2_temperature_difference:.3f} ({r2_temperature_difference_numerator:.6f}/{r2_temperature_difference_denominator:.6f})")
    im6 = plot_top_layer_temperature(ax6, y_hat, scan_params, vmin=-100, vmax=100, cmap='seismic')

    ax7.set_title("Temperature Error")
    im7 = plot_top_layer_temperature(ax7, next_x_p - next_x_gt, scan_params, vmin=-1, vmax=1, cmap='seismic')

    ax8.set_title("Temperature Difference Error")
    im8 = plot_top_layer_temperature(ax8, y_hat - y, scan_params, vmin=-100, vmax=100, cmap='seismic')

    ims = [im1, im2, im3, im4, im5, im6, im7, im8]

    if setup:
        for ax, im in zip(axes.flatten(), ims):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

    return [im1, im2, im3, im4, im5, im6, im7, im8]


frames_step = 5
num_frames = (len(x_gt_values) - 1) // frames_step
t_start = 0

ani = matplotlib.animation.FuncAnimation(fig, lambda t: plot_model_at_timestep(t_start + frames_step + t*frames_step), frames=num_frames, blit=True, interval=500, repeat_delay=1000, init_func=lambda: plot_model_at_timestep(t_start, setup=True))

FFwriter = matplotlib.animation.FFMpegWriter(fps=2)
animation_filepath = f"./plots/{model_name}_full_rolout_5_step.mp4"
ani.save(animation_filepath, writer=FFwriter)
