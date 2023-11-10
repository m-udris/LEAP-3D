import torch


def TransformIncorrectShapeError(Exception):
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

    raise TransformIncorrectShapeError()

def _normalize_min_max(x, min_value, max_value, inverse=False):
    if inverse:
        return x * (max_value - min_value) + min_value
    return (x - min_value) / (max_value - min_value)

# ['scanning_angle', 'laser_power', 'laser_radius']
def normalize_extra_param(x, index, min_value=0, max_value=1, inplace=False, inverse=False):
    if not inplace:
        x = x.clone()

    # If no window
    if len(x.shape) == 1:
        x[index] = _normalize_min_max(x[index], min_value, max_value, inverse=inverse)
        return x
    # If window
    if len(x.shape) == 3:
        x[:, index] = _normalize_min_max(x[:, index], min_value, max_value, inverse=inverse)
        return x

    raise TransformIncorrectShapeError()


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

    raise TransformIncorrectShapeError()
