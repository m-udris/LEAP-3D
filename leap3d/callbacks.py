import lightning.pytorch as pl
import torch
from torchmetrics.functional import r2_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb


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