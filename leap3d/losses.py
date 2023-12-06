import torch


def heat_loss(y_hat, y, eps=0.05):
    max_diff = torch.max(torch.max(torch.abs(y), dim=-1).values, dim=-1, keepdims=True).values
    new_eps = max_diff * eps
    return torch.mean((y_hat - y)**2 * (torch.abs(y) + new_eps))