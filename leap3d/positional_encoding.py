import torch

def positional_encoding(p, L=3):
    pi_times_p = torch.pi * p
    powers_L = torch.pow(2, torch.arange(L, requires_grad=False, device=p.device, dtype=p.dtype))
    pi_times_p = pi_times_p.unsqueeze(-1)
    sin_values = torch.sin(powers_L * pi_times_p)
    cos_values = torch.cos(powers_L * pi_times_p)

    output = torch.cat((sin_values, cos_values), dim=-1)
    assert output.shape == (*p.shape, 2 * L)
    if len(p.shape) == 1:
        output = output.squeeze(0)
    return output
