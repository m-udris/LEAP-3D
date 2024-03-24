import torch
from torch import nn

def heat_loss(y_hat, y, eps=0.05):
    max_diff = torch.max(torch.max(torch.abs(y), dim=-1).values, dim=-1, keepdims=True).values
    new_eps = max_diff * eps
    return torch.mean((y_hat - y)**2 * (torch.abs(y) + new_eps))


def weighted_l1_loss(y_hat, y):
    w = 1.25 - torch.abs(1 - y)
    loss = w * nn.functional.l1_loss(y_hat, y, reduction='none')
    return torch.mean(loss)


def heavy_weighted_l1_loss(y_hat, y):
    w = 1 - torch.abs(1 - y)**2
    loss = w * nn.functional.l1_loss(y_hat, y, reduction='none')
    return torch.mean(loss)


def distance_l1_loss_2d(y_hat, y, distances):
    w = 1.25 - distances / torch.max(distances)
    loss = w * nn.functional.l1_loss(y_hat, y, reduction='none')
    return torch.mean(loss)


# https://github.com/pytorch/pytorch/issues/12327
def smooth_l1_loss(input, target, beta=1, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()