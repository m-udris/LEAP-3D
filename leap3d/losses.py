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
