import torch

from leap3d.config import ROUGH_COORDS_STEP_SIZE, TIMESTEP_DURATION


def get_positional_gradients(x, spacing=ROUGH_COORDS_STEP_SIZE):
    return torch.gradient(torch.tensor(x), spacing=spacing)

def get_temporal_gradients(x_t1, x_t2, timestep_duration=TIMESTEP_DURATION):
    x_t1 = torch.tensor(x_t1)
    x_t2 = torch.tensor(x_t2)
    x = torch.stack([x_t1, x_t2], dim=0)
    gradients = torch.gradient(x, spacing=timestep_duration)
    return gradients[0]
