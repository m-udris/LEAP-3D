import torch

from leap3d.config import ROUGH_COORDS_STEP_SIZE, TIMESTEP_DURATION


def get_positional_gradients(x):
    return torch.gradient(torch.tensor(x), spacing=ROUGH_COORDS_STEP_SIZE)

def get_temporal_gradients(x_t1, x_t2):
    x = torch.stack([x_t1, x_t2], dim=0)
    gradients = torch.gradient(x, spacing=TIMESTEP_DURATION)
    return gradients[0]
