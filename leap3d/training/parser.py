import yaml

import torch
from torchvision import transforms

from leap3d.config import BASE_TEMPERATURE, MELTING_POINT
from leap3d.dataset import ExtraParam
from leap3d.transforms import normalize_temperature_2d


def parse_normalize_temperature_2d(transform_config, min_temperature=BASE_TEMPERATURE, max_temperature=MELTING_POINT):
    base_temperature = transform_config.get('base_temperature', min_temperature)
    if base_temperature == 'base':
        base_temperature = BASE_TEMPERATURE
    melting_point = transform_config.get('melting_point', max_temperature)
    if melting_point == 'melting_point':
        melting_point = MELTING_POINT
    transform = transforms.Lambda(
        lambda x: normalize_temperature_2d(x, base_temperature=base_temperature, melting_point=melting_point, inplace=True))
    return transform

def parse_train_transform(train_transform_config):
    transform_list = []
    for transform_config in train_transform_config:
        if transform_config.name == 'tensor':
            transform = torch.tensor
        elif transform_config.name == 'normalize_temperature_2d':
            transform = parse_normalize_temperature_2d(transform_config)
        else:
            raise ValueError(f'Unknown transform name: {transform_config.name}')

        transform_list.append(transform)

    return transforms.Compose(transform_list)


def parse_target_transform(target_transform_config):
    transforms_list = []
    for transform_config in target_transform_config:
        if transform_config.name == 'tensor':
            transform = torch.tensor
        elif transform_config.name == 'normalize_temperature_2d':
            transform = parse_normalize_temperature_2d(transform_config, min_temperature=0, max_temperature=10)
        else:
            raise ValueError(f'Unknown transform name: {transform_config.name}')

        transforms_list.append(transform)

    return transforms.Compose(transforms_list)


def parse_extra_params(extra_params_config):
    return [ExtraParam(extra_param_config.name) for extra_param_config in extra_params_config]


def parse_extra_params_transform(extra_params_transform_config):
    transform_list = []
    for transform_config in extra_params_transform_config:
        if transform_config.name == 'tensor':
            transform = torch.tensor4
        # TODO
        else:
            raise ValueError(f'Unknown transform name: {transform_config.name}')

        transform_list.append(transform)

    return transforms.Compose(transform_list)

def parse_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)



    return config