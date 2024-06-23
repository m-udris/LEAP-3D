import torch


class TransformIncorrectShapeError(Exception):
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f'TransformIncorrectShapeError: {self.shape}'
    pass


def normalize_temperature_2d(x, melting_point, base_temperature=0, temperature_channel_index=-1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no channel
    if len(x.shape) == 2:
        x = _normalize_min_max(x, base_temperature, melting_point, inverse=inverse)
        return x
    # If no window
    if len(x.shape) == 3:
        x[temperature_channel_index, :, :] = _normalize_min_max(x[temperature_channel_index, :, :], base_temperature, melting_point, inverse=inverse)
        return x
    # If window
    if len(x.shape) == 4:
        x[:, temperature_channel_index, :, :] = _normalize_min_max(x[:, temperature_channel_index, :, :], base_temperature, melting_point, inverse=inverse)
        return x

    raise TransformIncorrectShapeError(x.shape)

def normalize_temperature_3d(x, melting_point, base_temperature=0, temperature_channel_index=-1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no channel
    if len(x.shape) == 3:
        x = _normalize_min_max(x, base_temperature, melting_point, inverse=inverse)
        return x
    # If no window
    if len(x.shape) == 4:
        x[temperature_channel_index] = _normalize_min_max(x[temperature_channel_index], base_temperature, melting_point, inverse=inverse)
        return x
    # If window
    if len(x.shape) == 5:
        x[:, temperature_channel_index] = _normalize_min_max(x[:, temperature_channel_index], base_temperature, melting_point, inverse=inverse)
        return x

    raise TransformIncorrectShapeError(x.shape)


def normalize_temperature_log_2d(x, min_value=0, max_value=1, temperature_channel_index=-1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no channel
    if len(x.shape) == 2:
        x = _normalize_log(x, min_value, max_value, inverse=inverse)
        return x

    x[..., temperature_channel_index, :, :] = _normalize_log(x[..., temperature_channel_index, :, :], min_value, max_value, inverse=inverse)
    return x


def get_target_to_train_transform(train_min_value, train_max_value, target_min_value, target_max_value, simplified=True):
    def transform_simplified(x, inplace=False, inverse=False):
        if not inplace:
            x = x.clone()
        if inverse:
            x = (x * (train_max_value - train_min_value)) / (target_max_value - target_min_value)
            # x = (x * t2_inverse) - t1
            return x
        x = x * (target_max_value - target_min_value) / (train_max_value - train_min_value)
        return x

    def transform(x, inplace=False, inverse=False):
        if not inplace:
            x = x.clone()
        if inverse:
            x = _normalize_min_max(x, train_min_value, train_max_value, inverse=True)
            x = _normalize_min_max(x, target_min_value, target_max_value, inverse=False)
            return x

        x = _normalize_min_max(x, target_min_value, target_max_value, inverse=True)
        x = _normalize_min_max(x, train_min_value, train_max_value, inverse=False)
        return x

    return transform_simplified if simplified else transform


def get_target_log_to_train_transform(train_min_value, train_max_value, target_min_value, target_max_value):
    def transform(x, inplace=False, inverse=False):
        if not inplace:
            x = x.clone()
        if inverse:
            x = _normalize_min_max(x, train_min_value, train_max_value, inverse=True)
            x = _normalize_log(x, target_min_value, target_max_value, inverse=False)
            return x

        x = _normalize_log(x, target_min_value, target_max_value, inverse=True)
        x = _normalize_min_max(x, train_min_value, train_max_value, inverse=False)
        return x

    return transform



def _normalize_min_max(x, min_value, max_value, inverse=False):
    if inverse:
        return x * (max_value - min_value) + min_value
    return (x - min_value) / (max_value - min_value)


def _normalize_log(x, min_value, max_value, inverse=False):
    if inverse:
        return (torch.exp(x) * (max_value - min_value)) - 1 + min_value
    x = (torch.max(torch.zeros_like(x), (x - min_value)) + 1) / (max_value - min_value)
    if torch.any(x <= 0):
        print('Negative value in log normalization')
        raise ValueError
    return torch.log(x)


# ['scanning_angle', 'laser_power', 'laser_radius']
def normalize_extra_param(x, index, min_value=0, max_value=1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no window
    if len(x.shape) == 1:
        x[index] = _normalize_min_max(x[index], min_value, max_value, inverse=inverse)
        return x
    # If window
    if len(x.shape) == 2:
        x[:, index] = _normalize_min_max(x[:, index], min_value, max_value, inverse=inverse)
        return x

    raise TransformIncorrectShapeError(x.shape)


def cos_transform(x, inverse=False):
    if inverse:
        return torch.acos(x)
    return torch.cos(x)


def scanning_angle_cos_transform(x, index, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no window
    if len(x.shape) == 1:
        x[index] = cos_transform(x[index], inverse=inverse)
        return x
    # If window
    if len(x.shape) == 2:
        x[:, index] = cos_transform(x[:, index], inverse=inverse)
        return x

    raise TransformIncorrectShapeError(x.shape)


def normalize_positional_grad(x, index, norm_constant, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    if inverse:
        x[..., index] = x[..., index] * norm_constant
    else:
        x[..., index] = x[..., index] / norm_constant

    return x

def normalize_temporal_grad(x, index, max_temp, min_temp, norm_constant=1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    factor = norm_constant * (max_temp - min_temp)

    if inverse:
        x[..., index] = x[..., index] * factor
    else:
        x[..., index] = x[..., index] / factor

    return x
