from pathlib import Path
import lightning.pytorch as pl
from matplotlib import animation
import torch
from torchmetrics.functional import r2_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb

from leap3d.plotting import plot_model_top_layer_temperature_comparison


class LogR2ScoreOverTimePlotCallback(Callback):
    def __init__(self, steps=10, samples=100):
        super().__init__()
        self.steps = steps
        self.samples = samples

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_plot_r2_score_over_time(trainer, pl_module)

    def log_plot_r2_score_over_time(self, trainer, model):
        evaluation_dataset = trainer.datamodule.leap_eval
        current_epoch = trainer.current_epoch
        r2_score_data = self.get_r2_score_over_time(model, evaluation_dataset)

        table = wandb.Table(data=r2_score_data, columns = ["step", "R2 score"])
        wandb.log(
            {f"R2 score over {self.steps} steps after epoch {current_epoch}" : wandb.plot.line(table, "step", "R2 score",
                title=f"R2 score over {self.steps} steps after epoch {current_epoch}")})

    def get_r2_score_over_time(self, model, dataset):
        r2_score_values = []

        y_values = [torch.empty(0).to(model.device) for _ in range(self.steps)]
        y_hat_values = [torch.empty(0).to(model.device) for _ in range(self.steps)]

        for sample_idx in range(0, len(dataset) - self.steps, (len(dataset) - self.steps) // self.samples):
            next_x_temperature = None
            for i in range(self.steps):
                x, extra_params, y = dataset[sample_idx + i]
                x = x[0].clone().detach().to(model.device)
                extra_params = extra_params.clone().detach().to(model.device)
                y = y[0].clone().detach().to(model.device)

                if next_x_temperature is not None:
                    x[-1, :, :] = next_x_temperature
                y_hat = model(x, extra_params=extra_params)

                y_values[i] = torch.cat((y_values[i], y.reshape(-1)), dim=0)
                y_hat_values[i] = torch.cat((y_hat_values[i], y_hat.reshape(-1)), dim=0)
                next_x_temperature = model.get_predicted_temperature(x[-1], y_hat[0, :, :])

        for i, (y_value, y_hat_value) in enumerate(zip(y_values, y_hat_values)):
            r2_score_value = r2_score(y_hat_value, y_value)
            r2_score_values.append([i, r2_score_value])

        return r2_score_values


def get_checkpoint_only_last_epoch_callback(checkpoint_filename):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./model_checkpoints/",
        filename=checkpoint_filename, monitor="step", mode="max",
        save_top_k=1
    )
    return checkpoint_callback


class PlotTopLayerTemperatureCallback(Callback):
    def __init__(self, scan_parameters, plot_dir, steps=10, samples=100):
        super().__init__()
        self.scan_parameters = scan_parameters
        self.plot_dir = Path(plot_dir)
        self.steps = steps
        self.samples = samples

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_plot_top_layer_temperature(trainer, pl_module)

    def log_plot_top_layer_temperature(self, trainer, model):
        evaluation_dataset = trainer.datamodule.leap_eval
        current_epoch = trainer.current_epoch
        fig, axes, ims = plot_model_top_layer_temperature_comparison(self.scan_parameters, model, evaluation_dataset, self.steps, self.samples)

        ani = animation.ArtistAnimation(fig, ims, repeat=True, interval=500, blit=True, repeat_delay=1000)
        animation_filepath = self.plot_dir / f"top_layer_temperature_after_epoch_{current_epoch}.gif"
        ani.save(animation_filepath, fps=2, progress_callback=lambda x, _: print(x, end="\r"))
        wandb.log({"video": wandb.Video(str(animation_filepath), fps=4, format="gif")})


class PlotErrorOverTimeCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_plot_error_over_time(trainer, pl_module)

    def log_plot_error_over_time(self, trainer, model):
        evaluation_dataset = trainer.datamodule.leap_eval
        current_epoch = trainer.current_epoch
        error_data = self.get_error_over_time(model, evaluation_dataset)

        table = wandb.Table(data=error_data, columns = ["step", "abs_error", "rel_error", "r2_score"])
        wandb.log(
            {f"Relative error over time after epoch {current_epoch}" : wandb.plot.line(table, "step", "rel_error",
                title=f"Relative score over time after epoch {current_epoch}")})
        wandb.log(
            {f"Absolute error over time after epoch {current_epoch}" : wandb.plot.line(table, "step", "abs_error",
                title=f"Absolute error over time after epoch {current_epoch}")})
        wandb.log(
            {f"R2 over time after epoch {current_epoch}" : wandb.plot.line(table, "step", "r2_score",
                title=f"R2 over time after epoch {current_epoch}")})

    def get_error_over_time(self, model, dataset):
        error_data = []
        absolute_error_sum = 0
        relative_error_sum = 0
        r2_score_sum = 0
        count = 0

        model.eval()
        next_x_temperature = None
        for sample_idx in range(len(dataset)):
            x, extra_params, y = dataset[sample_idx]
            x = x[0].clone().detach().to(model.device)
            extra_params = extra_params.clone().detach().to(model.device)
            y = y[0].clone().detach().to(model.device)

            actual_x_temperature = x[-1]

            if next_x_temperature is not None:
                x[-1] = next_x_temperature
            y_hat = model(x, extra_params=extra_params)

            next_x_temperature = model.get_predicted_temperature(x[-1], y_hat[0])
            actual_next_x_temperature = model.get_predicted_temperature(actual_x_temperature, y[0])

            # e(t) = sum((gt[t,:,:]-pred[t,:,])**2) / mean_t(sum((gt[t,:,:])**2))
            relative_error = torch.sum((actual_next_x_temperature - next_x_temperature)**2) / torch.max(torch.mean(actual_next_x_temperature**2), torch.tensor(1e-6))
            relative_error = relative_error.detach().cpu().numpy()
            absolute_error = torch.sum(torch.abs(actual_next_x_temperature - next_x_temperature)) / torch.max(torch.mean(actual_next_x_temperature), torch.tensor(1e-6))
            absolute_error = absolute_error.detach().cpu().numpy()

            r2_score_value = r2_score(next_x_temperature, actual_next_x_temperature)

            relative_error_sum += relative_error
            absolute_error_sum += absolute_error
            r2_score_sum += r2_score_value

            count += 1

            if count % 100 == 0:
                average_relative_error = relative_error_sum / count
                average_absolute_error = absolute_error_sum / count
                average_r2_score = r2_score_sum / count
                error_data.append([sample_idx, average_absolute_error, average_relative_error, average_r2_score])
                average_relative_error = 0
                absolute_error_sum = 0
                relative_error_sum = 0
                count = 0

        if count > 0:
            average_relative_error = relative_error_sum / count
            average_absolute_error = absolute_error_sum / count
            average_r2_score = r2_score_sum / count
            error_data.append([sample_idx, average_absolute_error, average_relative_error, average_r2_score])

        return error_data