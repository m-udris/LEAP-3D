import sys
from pathlib import Path

import torch
from torchvision.transforms import transforms

from leap3d.datasets import UnetInterpolationDataModule, ExtraParam
from leap3d.config import DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, MAX_LASER_POWER, MAX_LASER_RADIUS, MELTING_POINT, BASE_TEMPERATURE, NUM_WORKERS, FORCE_PREPARE
from leap3d.transforms import normalize_extra_param, normalize_temperature_2d, normalize_temperature_3d, scanning_angle_cos_transform, get_target_to_train_transform


dataset_dir = Path(sys.argv[1])

train_transforms = {
    'input': transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True))
    ]),
    'target': transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inplace=True))
    ]),
    'extra_params': transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: scanning_angle_cos_transform(x, 0, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 1, 0, MAX_LASER_POWER, inplace=True)),
        transforms.Lambda(lambda x: normalize_extra_param(x, 2, 0, MAX_LASER_RADIUS, inplace=True))
    ])
}

inverse_transforms = {
    'input': transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True))
    ]),
    'target': transforms.Compose([
        torch.tensor,
        transforms.Lambda(lambda x: normalize_temperature_2d(x, melting_point=MELTING_POINT, base_temperature=BASE_TEMPERATURE, inverse=True, inplace=True))
    ])
}

dm = UnetInterpolationDataModule(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, DATA_DIR, dataset_dir,
                 is_3d=False, batch_size=32,
                 train_cases=18, test_cases=[18, 19],
                 extra_params=[ExtraParam.SCANNING_ANGLE, ExtraParam.LASER_POWER, ExtraParam.LASER_RADIUS],
                 transforms=train_transforms, inverse_transforms=inverse_transforms,
                 force_prepare=False, num_workers=0)

dm.prepare_data()
dm.setup()

tdl = dm.train_dataloader()

for input, extra_params, target in tdl:
    print(input.shape)
    print(extra_params.shape)
    print(target.shape)
    break