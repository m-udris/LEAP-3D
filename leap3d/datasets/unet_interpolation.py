from pathlib import Path
from typing import Callable, List, Dict

from leap3d.datasets import LEAPDataModule
from leap3d.datasets.channels import RoughTemperatureAroundLaser, TemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius


class UNetInterpolationDataModule(LEAPDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_directory: str="path/to/dir", prepared_data_path: str="path/to/file",
                 is_3d: bool=False,
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 input_shape: List[int]=[64, 64, 16], target_shape: List[int]=[64, 64, 16],
                 input_channels=[RoughTemperatureAroundLaser], extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
                 target_channels=[TemperatureAroundLaser],
                 transforms: Dict[str, Callable]={}, inverse_transforms: Dict[str, Callable]={},
                 force_prepare=False,
                 num_workers=0):
        super().__init__(
            scan_parameters_filepath=scan_parameters_filepath,
            rough_coordinates_filepath=rough_coordinates_filepath,
            raw_data_directory=raw_data_directory,
            prepared_data_path=prepared_data_path,
            is_3d=is_3d,
            batch_size=batch_size,
            train_cases=train_cases, test_cases=test_cases,
            input_shape=input_shape, target_shape=target_shape,
            input_channels=input_channels,
            extra_input_channels=extra_input_channels,
            target_channels=target_channels,
            transforms=transforms,
            inverse_transforms=inverse_transforms,
            force_prepare=force_prepare,
            num_workers=num_workers
        )
