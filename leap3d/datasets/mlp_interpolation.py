from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
import torch
from torch.utils.data import random_split

from leap3d.datasets import LEAPDataModule, LEAPDataset
from leap3d.datasets.channels import RoughTemperatureAroundLaser, TemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius


class MLPInterpolationDataset(LEAPDataset):
    def __init__(self, data_filepath: str="path/to/dir",
                 transforms: Dict={}, inverse_transforms: Dict={}):
        super().__init__(
            data_filepath=data_filepath,
            transforms=transforms,
            inverse_transforms=inverse_transforms
        )
        self.target_coordinates = self.data_file['target_coordinates']
        self.melting_pool = self.data_file['melting_pool']
        self.melting_pool_indices = self.data_file['melting_pool_indices']
        self.laser_data = self.data_file['laser_data']

        self.melting_pool_transform = self.transforms.get('melting_pool', lambda x: x)
        self.laser_data_transform = self.transforms.get('laser_data', lambda x: x)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        extra_input = self.extra_inputs[idx]
        target = self.targets[idx]

        input = self.input_transform(input)
        extra_input = self.extra_input_transform(extra_input)
        target = self.target_transform(target)

        target_coordinates = self.target_coordinates[idx]

        melting_pool_idx_start, melting_pool_idx_end = self.melting_pool_indices[idx]
        melting_pool = self.melting_pool[int(melting_pool_idx_start):int(melting_pool_idx_end)]
        laser_data = self.laser_data[idx]

        melting_pool = self.melting_pool_transform(melting_pool)
        laser_data = self.laser_data_transform(laser_data)

        return input, extra_input, target, target_coordinates, melting_pool, laser_data


class MLPInterpolationDataModule(LEAPDataModule):
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
        self.melting_pool_offset = 0
        self.melt_pool_shape = (8,)
        self.rough_coordinates_size = 16
        self.interpolated_coordinates_scale = 0.25

    def create_h5_datasets(self, h5py_file):
        datasets = super().create_h5_datasets(h5py_file)
        datasets['target_coordinates'] = self.create_h5_target_coordinates_dataset(h5py_file)
        datasets['melting_pool'] = self.create_h5_melting_pool_dataset(h5py_file)
        datasets['melting_pool_indices'] = self.create_h5_melting_pool_indices_dataset(h5py_file)
        datasets['laser_data'] = self.create_h5_laser_data_dataset(h5py_file)

        return datasets

    def create_h5_target_coordinates_dataset(self, h5py_file):
        dset_shape = [1, 6 if self.is_3d else 4]  # x_min, x_max, y_min, y_max, [z_min, z_max]
        dset_maxshape = [None, *dset_shape[1:]]

        return h5py_file.create_dataset("target_coordinates", dset_shape, maxshape=dset_maxshape, chunks=True)

    def create_h5_melting_pool_dataset(self, h5py_file):
        dset_shape = (1, 8)  # x, y, z, temperature, temperature gradient x, temperature gradient y, temperature gradient z, phase indicator
        dset_maxshape = (None, *dset_shape[1:])

        return h5py_file.create_dataset("melting_pool", dset_shape, maxshape=dset_maxshape, chunks=True)

    def create_h5_melting_pool_indices_dataset(self, h5py_file):
        dset_shape = (1, 2)  # start index, end index
        dset_maxshape = (None, 2)

        return h5py_file.create_dataset("melting_pool_indices", dset_shape, maxshape=dset_maxshape, chunks=True)

    def create_h5_laser_data_dataset(self, h5py_file):
        dset_shape = (1, 7)  # x position, y position, laser power, laser radius, laser velocity x, laser velocity y, existence of melt pool
        dset_maxshape = (None, 7)

        return h5py_file.create_dataset("laser_data", dset_shape, maxshape=dset_maxshape, chunks=True)

    def get_data_generator_timestep(self, scan_results, scan_parameters, timestep):
        data = super().get_data_generator_timestep(scan_results, scan_parameters, timestep)
        data['target_coordinates'] = self.get_target_coordinates(scan_results, scan_parameters, timestep)
        melting_pool, melting_pool_indices = self.get_melting_pool(scan_results, timestep)
        data['melting_pool'] = melting_pool
        data['melting_pool_indices'] = melting_pool_indices
        data['laser_data'] = self.get_laser_data(scan_results, timestep)

        return data

    def get_target_coordinates(self, scan_results, scan_parameters, timestep):
        laser_x, laser_y = scan_results.laser_data[timestep][:2]
        bounds = scan_parameters.get_bounds_around_position(laser_x, laser_y, self.rough_coordinates_size)
        if self.is_3d:
            return np.array(bounds)
        return np.array(bounds[:4])

    def get_melting_pool(self, scan_results, timestep):
        melt_pool_data = scan_results.melt_pool[timestep]
        if melt_pool_data == [[]]:
            melt_pool_data = np.zeros((0, *self.melt_pool_shape))
        else:
            melt_pool_data = np.array(melt_pool_data)

            # filter out points that are not in the last layer if 2D
            if not self.is_3d:
                max_z = np.max(melt_pool_data[:, 2])
                melt_pool_data = melt_pool_data[melt_pool_data[:, 2] == max_z]

        n_points = melt_pool_data.shape[0]

        melting_pool_indices = np.array([self.melting_pool_offset, self.melting_pool_offset + n_points])
        self.melting_pool_offset += n_points

        return melt_pool_data, melting_pool_indices

    def get_laser_data(self, scan_results, timestep):
        laser_data = np.array(scan_results.laser_data[timestep])

        return laser_data

    def write_to_h5_dataset(self, dset, data, offset):
        if dset.name == '/melting_pool':
            flattened_data = []
            for item in data:
                flattened_data.extend(item)
            data = np.array(flattened_data)
            dset.resize(self.melting_pool_offset + data.shape[0], axis=0)
            dset[self.melting_pool_offset:] = data
            self.melting_pool_offset += data.shape[0]
        else:
            data = np.array(data)
            dset.resize(offset + data.shape[0], axis=0)
            dset[offset:] = data
        dset.flush()

    def setup(self, stage: str, split_ratio: float=0.8):
        if stage == 'fit' or stage is None:
            full_dataset_path = self.prepared_data_path / "dataset.hdf5"
            full_dataset = MLPInterpolationDataset(full_dataset_path, transforms=self.transforms, inverse_transforms=self.inverse_transforms)

            train_points_count = int(np.ceil(len(full_dataset) * split_ratio))
            val_points_count = len(full_dataset) - train_points_count
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
            )

        if stage == 'test' or stage is None:
            test_dataset_path = self.prepared_data_path / "test_dataset.hdf5"
            self.test_dataset = MLPInterpolationDataset(test_dataset_path, transforms=self.transforms, inverse_transforms=self.inverse_transforms)
