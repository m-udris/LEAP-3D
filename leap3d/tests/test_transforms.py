import numpy as np
import torch

from leap3d.config import MELTING_POINT, BASE_TEMPERATURE
from leap3d.transforms import get_target_log_to_train_transform, get_target_to_train_transform, normalize_temperature_2d, normalize_temperature_log_2d


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


def test_target_to_train_transform():
    for _ in range(10):
        x = torch.rand(100)

        train_min_value, train_max_value, target_min_value, target_max_value = np.random.rand(4)

        transform = get_target_to_train_transform(train_min_value, train_max_value, target_min_value, target_max_value, simplified=False)
        transform_simplified = get_target_to_train_transform(train_min_value, train_max_value, target_min_value, target_max_value, simplified=True)

        x_transformed = transform(x)
        x_transformed_simple = transform_simplified(x)

        assert torch.allclose(x_transformed, x_transformed_simple, atol=NORMALIZE_ATOL)

        x_transformed = transform(x_transformed, inverse=True)
        x_transformed_simple = transform_simplified(x_transformed_simple, inverse=True)

        assert torch.allclose(x_transformed, x, atol=NORMALIZE_ATOL)
        assert torch.allclose(x_transformed_simple, x, atol=NORMALIZE_ATOL)


def test_log_transforms():
    temperature_min = 300
    temperature_max = 3000
    temperature_difference_min = -750
    temperature_difference_max = 1750

    for _ in range(100):
        temperature = torch.rand(3, 64, 64) * (temperature_max - temperature_min) + temperature_min
        temperature_difference = torch.rand(1, 64, 64) * (temperature_difference_max - temperature_difference_min) + temperature_difference_min

        temperature_normalized = normalize_temperature_2d(temperature, temperature_max, temperature_min, inplace=False)
        temperature_difference_normalized = normalize_temperature_log_2d(temperature_difference, temperature_difference_min, temperature_difference_max, inplace=False)

        next_temperature_gt = temperature + temperature_difference
        next_temperature_gt_normalized = normalize_temperature_2d(next_temperature_gt, temperature_max, temperature_min, inplace=False)[-1]

        target_to_train_transform = get_target_log_to_train_transform(temperature_min, temperature_max, temperature_difference_min, temperature_difference_max)
        temperature_diff_transformed = target_to_train_transform(temperature_difference_normalized, inplace=False)

        next_temperature_normalized = temperature_normalized[-1] + temperature_diff_transformed[-1]

        assert torch.allclose(next_temperature_gt_normalized, next_temperature_normalized, atol=1e-7)
