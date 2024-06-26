from pathlib import Path
from typing import Callable, List
import lightning.pytorch as pl
from matplotlib import animation
import numpy as np
import torch
from torchmetrics.functional import r2_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb

from leap3d.plotting import plot_model_top_layer_temperature_comparison
from leap3d.config import DATASET_DIR

def get_recursive_model_predictions(model, dataset):
    model.eval()
    next_x_temperature = None
    with torch.no_grad():
        for index, (x, extra_params, y) in enumerate(dataset):
            if index >= 10000:
                break
            x = x[0].to(model.device)
            x_original = x[-1].clone().detach()
            extra_params = extra_params.to(model.device)
            y = y[0].to(model.device)

            if next_x_temperature is not None:
                x[-1] = next_x_temperature
            y_hat = model(x, extra_params=extra_params)[0]

            next_x_temperature = model.get_predicted_temperature(x[-1], y_hat)

            x_original = x_original.detach().cpu()
            y = y.squeeze().detach().cpu()
            y_hat = y_hat.squeeze().detach().cpu()
            next_x_temperature_copy = next_x_temperature.detach().cpu()
            yield x_original, y, y_hat, next_x_temperature_copy


def get_r2_scores(pred_list, target_list, return_parts=False):
    r2_scores = []
    numerators = []
    denominators = []
    for y_hat, y in zip(pred_list, target_list):
        if return_parts:
            r2_score_value, numerator, denominator = get_r2_score(y_hat, y, return_parts=True)
            r2_scores.append(r2_score_value)
            numerators.append(numerator)
            denominators.append(denominator)
        else:
            r2_score_value = get_r2_score(y_hat, y)
            r2_scores.append(r2_score_value)
    if return_parts:
        return r2_scores, numerators, denominators
    return r2_scores


def get_r2_score(y_hat, y, return_parts=False):
    r2_score_value = r2_score(y_hat.reshape(-1), y.reshape(-1))
    if return_parts:
        numerator = torch.sum((y_hat - y) ** 2)
        denominator = torch.sum((y - torch.mean(y)) ** 2)
        return r2_score_value, numerator, denominator
    return r2_score_value


# TODO: Refactor into a class
def get_relative_error(pred_list, target_list):
    relative_errors = []
    target_list_squares = np.array(target_list)**2
    target_list_temp_sums = np.sum(target_list_squares, axis=(1,2))
    target_list_mean = np.mean(target_list_temp_sums)

    for y_hat, y in zip(pred_list, target_list):
        relative_error = torch.sum((y_hat - y)**2) / target_list_mean
        relative_errors.append(relative_error)
    return relative_errors


# TODO: Refactor into a class
def get_absolute_error(pred_list, target_list):
    absolute_errors = []
    target_list_abs = np.abs(np.array(target_list))
    target_list_temp_sums = np.sum(target_list_abs, axis=(1,2))
    target_list_mean = np.mean(target_list_temp_sums)

    for y_hat, y in zip(pred_list, target_list):
        absolute_error = torch.sum(np.abs(y_hat - y)) / target_list_mean
        absolute_errors.append(absolute_error)
    return absolute_errors

def log_plot(plot_name, value_name, values):
    table = wandb.Table(data=[[t, value] for (t, value) in enumerate(values)], columns = ["step", value_name])
    wandb.log(
        {plot_name : wandb.plot.line(table, "step", value_name,
            title=plot_name)})


class LogR2ScoreOverTimePlotCallback(Callback):
    def __init__(self, steps=10, samples=100):
        super().__init__()
        self.steps = steps
        self.samples = samples

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for case_id, dataset in zip(trainer.datamodule.eval_cases, trainer.datamodule.eval_datasets):
            self.log_plot_r2_score_over_time(pl_module, dataset, case_id)

    def log_plot_r2_score_over_time(self, model, dataset, case_id):
        r2_score_data = self.get_r2_score_over_time(model, dataset)

        table = wandb.Table(data=r2_score_data, columns = ["step", "R2 score"])
        wandb.log(
            {f"R2 score over {self.steps} steps for case {case_id}" : wandb.plot.line(table, "step", "R2 score",
                title=f"R2 score over {self.steps} steps for case {case_id}")})

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


def get_checkpoint_only_last_epoch_callback(checkpoint_filename, monitor="step", mode="max", filename=None):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=DATASET_DIR / "model_checkpoints/", monitor=monitor, mode=mode,
        save_top_k=1, filename=filename
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
        for case_id, dataset in zip(trainer.datamodule.eval_cases, trainer.datamodule.eval_datasets):
            self.log_plot_top_layer_temperature(trainer, pl_module, dataset, case_id)

    def log_plot_top_layer_temperature(self, trainer, model, dataset, case_id):

        current_epoch = trainer.current_epoch
        fig, axes, ims = plot_model_top_layer_temperature_comparison(self.scan_parameters, model, dataset, self.steps, self.samples)

        ani = animation.ArtistAnimation(fig, ims, repeat=True, interval=500, blit=True, repeat_delay=1000)
        animation_filepath = self.plot_dir / f"top_layer_temperature_for_case_{case_id}_after_epoch_{current_epoch}.gif"
        ani.save(animation_filepath, fps=2, progress_callback=lambda x, _: print(x, end="\r"))
        wandb.log({"video": wandb.Video(str(animation_filepath), fps=4, format="gif")})


class PlotErrorOverTimeCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for case_id, dataset in zip(trainer.datamodule.eval_cases, trainer.datamodule.eval_datasets):
            self.log_plot_error_over_time(pl_module, dataset, case_id)

    def log_plot_error_over_time(self, model, dataset, case_id):
        error_data = self.get_error_over_time(model, dataset)

        table = wandb.Table(data=error_data, columns = ["step", "abs_error", "rel_error", "r2_score"])
        wandb.log(
            {f"Relative error over time for case {case_id}" : wandb.plot.line(table, "step", "rel_error",
                title=f"Relative score over time for case {case_id}")})
        wandb.log(
            {f"Absolute error over time for case {case_id}" : wandb.plot.line(table, "step", "abs_error",
                title=f"Absolute error over time for case {case_id}")})
        wandb.log(
            {f"R2 over time for case {case_id}" : wandb.plot.line(table, "step", "r2_score",
                title=f"R2 over time for case {case_id}")})

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
            # print(dataset[sample_idx + 1][0][0, -1] - actual_next_x_temperature)
            # e(t) = sum((gt[t,:,:]-pred[t,:,])**2) / mean_t(sum((gt[t,:,:])**2))
            relative_error = torch.sum((actual_next_x_temperature - next_x_temperature)**2) / torch.max(torch.mean(actual_next_x_temperature**2), torch.tensor(1e-6))
            relative_error = relative_error.detach().cpu().numpy()
            absolute_error = torch.sum(torch.abs(actual_next_x_temperature - next_x_temperature)) / torch.max(torch.mean(actual_next_x_temperature), torch.tensor(1e-6))
            absolute_error = absolute_error.detach().cpu().numpy()

            r2_score_value = r2_score(y_hat.reshape(-1), y.reshape(-1))

            relative_error_sum += relative_error
            absolute_error_sum += absolute_error
            r2_score_sum += r2_score_value

            count += 1

            if count % 100 == 0:
                average_relative_error = relative_error_sum / count
                average_absolute_error = absolute_error_sum / count
                average_r2_score = r2_score_sum / count
                error_data.append([sample_idx, average_absolute_error, average_relative_error, average_r2_score])
                absolute_error_sum = 0
                relative_error_sum = 0
                r2_score_sum = 0
                count = 0

        if count > 0:
            average_relative_error = relative_error_sum / count
            average_absolute_error = absolute_error_sum / count
            average_r2_score = r2_score_sum / count
            error_data.append([sample_idx, average_absolute_error, average_relative_error, average_r2_score])

        return error_data


class Rollout2DUNetCallback(Callback):
    def __init__(self, scan_parameters, steps=10, samples=100):
        super().__init__()
        self.scan_parameters = scan_parameters
        self.steps = steps
        self.samples = samples

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if current_epoch < 10:
            return
        if current_epoch < 25 and current_epoch % 5 != 0:
            return

        for case_id, dataset in zip(trainer.datamodule.eval_cases, trainer.datamodule.eval_datasets):
            self.rollout(pl_module, dataset, case_id, current_epoch)

    def rollout(self, pl_module, dataset, case_id, current_epoch):
        predictions = get_recursive_model_predictions(pl_module, dataset)

        previous_x_pred_value = None

        temperature_diff_r2_scores = []
        temperature_r2_scores = []
        relative_errors = []
        relative_error_normalizer_list = []
        absolute_errors = []
        absolute_error_normalizer_list = []

        with torch.no_grad():
            for x, y, y_hat, x_pred_temperature in predictions:
                if previous_x_pred_value is not None:
                    temperature_r2_scores.append(get_r2_score(previous_x_pred_value, x))

                    relative_errors.append(torch.sum((previous_x_pred_value - x)**2))
                    relative_error_normalizer_list.append(np.sum(x.numpy()**2))

                    absolute_errors.append(torch.sum(np.abs(previous_x_pred_value - x)))
                    absolute_error_normalizer_list.append(np.sum(np.abs(x.numpy())))

                temperature_diff_r2_scores.append(get_r2_score(y_hat, y))


                previous_x_pred_value = x_pred_temperature

            relative_error_normalizer = np.mean(np.array(relative_error_normalizer_list))
            relative_errors = [relative_error / relative_error_normalizer for relative_error in relative_errors]

            absolute_error_normalizer = np.mean(np.array(absolute_error_normalizer_list))
            absolute_errors = [absolute_error / absolute_error_normalizer for absolute_error in absolute_errors]

        log_plot(f"Temperature Diff R2 score for rollout, case {case_id}, epoch {current_epoch}", "R2 Score", temperature_diff_r2_scores)
        log_plot(f"Temperature R2 score for rollout, case {case_id}, epoch {current_epoch}", "R2 Score", temperature_r2_scores)
        log_plot(f"Relative error for rollout, case {case_id}, epoch {current_epoch}", "Relative error", relative_errors)
        log_plot(f"Absolute error for rollout, case {case_id}, epoch {current_epoch}", "Absolute error", absolute_errors)

    def calculate_and_log_r2(self, x_gt_values, y_values, x_pred_values, y_hat_values, case_id):
        r2_scores = get_r2_scores(y_hat_values, y_values)
        log_plot(f"Temperature Diff R2 score for rollout, case {case_id}", "R2 Score", r2_scores)

        r2_scores = get_r2_scores(x_pred_values, x_gt_values)
        log_plot(f"Temperature R2 score for rollout, case {case_id}", "R2 Score", r2_scores)

    def calculate_and_log_relative_error(self, x_gt_values, x_pred_values, case_id):
        relative_errors = get_relative_error(x_pred_values, x_gt_values)
        log_plot(f"Relative error for rollout, case {case_id}", "Relative error", relative_errors)

    def calculate_and_log_absolute_error(self, x_gt_values, x_pred_values, case_id):
        absolute_errors = get_absolute_error(x_pred_values, x_gt_values)
        log_plot(f"Absolute error for rollout, case {case_id}", "Absolute error", absolute_errors)
