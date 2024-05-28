from typing import Callable, List, Dict
from pathlib import Path

import numpy as np
import shapely.geometry as sg
import torch
from torch.utils.data import random_split

from leap3d.contour import get_melting_pool_contour_2d
from leap3d.datasets import LEAPDataModule, LEAPDataset
from leap3d.datasets.channels import RoughTemperature, RoughTemperatureDifference, ScanningAngle, LaserPower, LaserRadius
from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.scanning.scan_results import ScanResults


class UNetForecastingDataset(LEAPDataset):
    def __init__(self, data_filepath: str="path/to/dir",
                 transforms: Dict={}, inverse_transforms: Dict={}):
        super().__init__(
            data_filepath=data_filepath,
            transforms=transforms,
            inverse_transforms=inverse_transforms
        )


class UNetForecastingDataModule(LEAPDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_directory: str="path/to/dir", prepared_data_path: str="path/to/file",
                 is_3d: bool=False,
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None, eval_cases: int | List=None,
                 input_shape: List[int]=[64, 64, 16], target_shape: List[int]=[64, 64, 16],
                 input_channels=[RoughTemperature], extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
                 target_channels=[RoughTemperatureDifference],
                 transforms: Dict[str, Callable]={}, inverse_transforms: Dict[str, Callable]={},
                 window_size=5,
                 window_step_size=5,
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
        self.dataset_class = UNetForecastingDataset

        if isinstance(eval_cases, int):
            self.eval_cases = list(range(eval_cases))
        else:
            self.eval_cases = eval_cases

        self.window_size = window_size
        self.window_step_size = window_step_size

    def create_h5_input_dataset(self, h5py_file):
        input_dset_shape = [1, self.window_size, self.input_channel_len, *self.input_shape]
        input_dset_maxshape = [None, *input_dset_shape[1:]]

        return h5py_file.create_dataset("input", input_dset_shape, maxshape=input_dset_maxshape, chunks=True)

    def create_h5_extra_input_dataset(self, h5py_file):
        extra_input_dset_shape = [1, self.window_size, self.extra_input_channel_len]
        extra_input_dset_maxshape = [None, *extra_input_dset_shape[1:]]

        return h5py_file.create_dataset("extra_input", extra_input_dset_shape, maxshape=extra_input_dset_maxshape, chunks=True)

    def create_h5_target_dataset(self, h5py_file):
        target_dset_shape = (1, self.window_size, self.target_channel_len, *self.target_shape)
        target_dset_maxshape = (None, *target_dset_shape[1:])

        return h5py_file.create_dataset("target", target_dset_shape, maxshape=target_dset_maxshape, chunks=True)

    def get_data_generator(self, case_id, scan_result_filepath, params_file, rough_coordinates):
        scan_results = ScanResults(scan_result_filepath)
        scan_parameters = ScanParameters(params_file, rough_coordinates, case_id)

        for timestep in range(0, scan_results.total_timesteps - self.window_size, self.window_step_size):
            yield self.get_data_generator_timestep(scan_results, scan_parameters, timestep)

    def get_data_generator_timestep(self, scan_results, scan_parameters, timestep):
        input_window = []
        extra_input_window = []
        target_window = []

        for t in range(timestep, timestep + self.window_size):
            input_channels = []
            extra_input_channels = []
            target_channels = []

            for channel in self.input_channels:
                input_channels.extend(
                    channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=t))

            for channel in self.extra_input_channels:
                extra_input_channels.extend(
                    channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=t))

            for channel in self.target_channels:
                target_channels.extend(
                    channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=t))

            input_window.append(input_channels)
            extra_input_window.append(extra_input_channels)
            target_window.append(target_channels)

        return {
            "input": input_window,
            "extra_input": extra_input_window,
            "target": target_window
        }

    def setup(self, stage: str, split_ratio: float=0.8):
        super().setup(stage, split_ratio)

        # eval data
        if stage == 'fit' or stage is None:
            self.eval_datasets = []

            for case_id in self.eval_cases:
                eval_dataset_path = self.prepared_data_path / f"eval_case_{case_id}/dataset.hdf5"
                eval_dataset = self.dataset_class(eval_dataset_path, transforms=self.transforms, inverse_transforms=self.inverse_transforms)
                self.eval_datasets.append(eval_dataset)
