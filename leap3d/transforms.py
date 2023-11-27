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


def get_target_to_train_transform(train_min_value, train_max_value, target_min_value, target_max_value, simplified=True):
    t1 = (target_min_value - train_min_value) / (target_max_value - target_min_value)
    t2 = (target_max_value - target_min_value) / (train_max_value - train_min_value)
    t2_inverse = (train_max_value - train_min_value) / (target_max_value - target_min_value)

    def transform_simplified(x, inplace=False, inverse=False):
        if not inplace:
            x = x.clone()
        if inverse:
            x = (x * t2_inverse) - t1
            return x
        x = (x + t1) * t2
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
