import torch

from leap3d.config import MELTING_POINT, BASE_TEMPERATURE
from leap3d.transforms import normalize_temperature_2d


NORMALIZE_ATOL = 1e-4

def _test_normalize_temperature_2d(x, melting_point, base_temperature=0, inplace=False):
    x_shape = x.shape
    x_new = normalize_temperature_2d(x, melting_point, base_temperature=base_temperature, inplace=inplace)
    if not inplace:
        x = x_new
    # Check that the output tensor has the same shape as the input tensor
    assert x_shape == x.shape

    x_new = normalize_temperature_2d(x, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=inplace, inverse=True)
    if not inplace:
        x = x_new
    # Check that the output tensor has the same shape as the input tensor
    assert x_shape == x.shape

    return x


def test_normalize_temperature_2d_batch():
    for _ in range(10):
        # Create a random tensor with shape (10, 3, 64, 64)
        x = torch.rand(10, 3, 64, 64)
        # Normalize the temperature of the tensor
        x_new = _test_normalize_temperature_2d(x, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=False)
        # Check that the unnormalized tensor is equal to the original tensor
        assert torch.allclose(x, x_new, atol=NORMALIZE_ATOL)


def test_normalize_temperature_2d_batch_inplace():
    for _ in range(10):
        # Create a random tensor with shape (10, 3, 64, 64)
        x = torch.rand(10, 3, 64, 64)
        x_new = x.clone()
        # Normalize the temperature of the tensor
        _test_normalize_temperature_2d(x_new, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True)
        # Check that the unnormalized tensor is equal to the original tensor
        assert torch.allclose(x, x_new, atol=NORMALIZE_ATOL)


def test_normalize_temperature_2d_sample():
    for _ in range(10):
        # Create a random tensor with shape (10, 3, 64, 64)
        x = torch.rand(3, 64, 64)
        # Normalize the temperature of the tensor
        x_new = _test_normalize_temperature_2d(x, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=False)
        # Check that the unnormalized tensor is equal to the original tensor
        assert torch.allclose(x, x_new, atol=NORMALIZE_ATOL)


def test_normalize_temperature_2d_sample_inplace():
    for _ in range(10):
        # Create a random tensor with shape (10, 3, 64, 64)
        x = torch.rand(3, 64, 64)
        x_new = x.clone()
        # Normalize the temperature of the tensor
        _test_normalize_temperature_2d(x_new, MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True)
        # Check that the unnormalized tensor is equal to the original tensor
        assert torch.allclose(x, x_new, atol=NORMALIZE_ATOL)
