from typing import Callable, List, Dict
from pathlib import Path

import numpy as np
import shapely.geometry as sg
import torch
from torch.utils.data import random_split

from leap3d.contour import get_melting_pool_contour_2d
from leap3d.datasets import LEAPDataModule, LEAPDataset
from leap3d.datasets.channels import RoughTemperatureAroundLaser, TemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius


class UNetInterpolationDataset(LEAPDataset):
    def __init__(self, data_filepath: str="path/to/dir",
                 transforms: Dict={}, inverse_transforms: Dict={},
                 include_distances_to_melting_pool: bool=False,
                 include_coordinates: bool=False):
        super().__init__(
            data_filepath=data_filepath,
            transforms=transforms,
            inverse_transforms=inverse_transforms
        )
        self.include_distances_to_melting_pool = include_distances_to_melting_pool
        if self.include_distances_to_melting_pool:
            self.distances_to_melting_pool = self.data_file['distances_to_melting_pool']
        self.include_coordinates = include_coordinates

    def __getitem__(self, idx):
        input = self.inputs[idx]
        if not self.include_coordinates:
            input = input[..., -1:, :, :]
        extra_input = self.extra_inputs[idx]
        target = self.targets[idx]

        input = self.input_transform(input)
        extra_input = self.extra_input_transform(extra_input)
        target = self.target_transform(target)

        if not self.include_distances_to_melting_pool:
            return input, extra_input, target

        distances_to_melting_pool = self.distances_to_melting_pool[idx]

        return input, extra_input, target, distances_to_melting_pool


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
                 include_distances_to_melting_pool: bool=False,
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
        self.include_distances_to_melting_pool = include_distances_to_melting_pool
        self.dataset_class = UNetInterpolationDataset
        self.rough_coordinates_box_size = 32
        self.interpolated_coordinates_scale = 0.25

    def create_h5_datasets(self, h5py_file):
        datasets = super().create_h5_datasets(h5py_file)
        if self.include_distances_to_melting_pool:
            datasets['distances_to_melting_pool'] = self.create_h5_distances_to_melting_pool_dataset(h5py_file)

        return datasets

    def create_h5_distances_to_melting_pool_dataset(self, h5py_file):
        dset_shape = (1, *self.target_shape)
        dset_maxshape = (None, *dset_shape[1:])

        return h5py_file.create_dataset("distances_to_melting_pool", dset_shape, maxshape=dset_maxshape, chunks=True)

    def get_data_generator_timestep(self, scan_results, scan_parameters, timestep):
        data = super().get_data_generator_timestep(scan_results, scan_parameters, timestep)
        if self.include_distances_to_melting_pool:
            data['distances_to_melting_pool'] = self.get_target_distances_to_melting_pool(scan_results, scan_parameters, timestep)

        return data

    def get_target_distances_to_melting_pool(self, scan_results, scan_parameters, timestep):
        if self.is_3d:
            raise not NotImplementedError("3D not implemented yet")
        return self.get_target_distances_to_melting_pool_2d(scan_results, scan_parameters, timestep)

    def get_target_distances_to_melting_pool_2d(self, scan_results, scan_parameters, timestep):
        if scan_results.is_melt_pool_empty(timestep):
            return self.get_target_distances_to_laser(scan_results, scan_parameters, timestep)
        temperature_high_res = scan_results.get_melt_pool_temperature_grid(scan_parameters, timestep)
        contours = get_melting_pool_contour_2d(temperature_high_res, top_k=1)[0]
        if len(contours) == 0:
            return self.get_target_distances_to_laser(scan_results, scan_parameters, timestep)
        contour = np.array(contours[0])
        # Rescale coordinates from unit distance into meters
        contour = ((contour - 128) / 128) * 0.0012
        pg = sg.Polygon(contour)

        laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
        points = scan_parameters.get_coordinates_around_position(laser_x, laser_y,
                                                                 self.rough_coordinates_box_size,
                                                                 scale=self.interpolated_coordinates_scale,
                                                                 is_3d=self.is_3d)
        distances = [np.abs(pg.exterior.distance(sg.Point(p[:2]))) for p in points]
        distances = np.array(distances).reshape(self.target_shape)
        return distances

    def get_target_distances_to_laser(self, scan_results, scan_parameters, timestep):
        laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
        laser_point = sg.Point(laser_x, laser_y)
        points_to_interpolate = scan_parameters.get_coordinates_around_position(laser_x, laser_y,
                                                                                self.rough_coordinates_box_size,
                                                                                scale=self.interpolated_coordinates_scale,
                                                                                is_3d=self.is_3d)
        distances = [np.abs(laser_point.distance(sg.Point(point))) for point in points_to_interpolate]
        distances = np.array(distances).reshape(self.target_shape)
        return distances

    def setup(self, stage: str, split_ratio: float=0.8):
        if stage == 'fit' or stage is None:
            full_dataset_path = self.prepared_data_path / "dataset.hdf5"
            full_dataset = self.dataset_class(
                full_dataset_path,
                transforms=self.transforms,
                inverse_transforms=self.inverse_transforms,
                include_distances_to_melting_pool=self.include_distances_to_melting_pool,
                include_coordinates=self.input_channels[0].return_coordinates)

            train_points_count = int(np.ceil(len(full_dataset) * split_ratio))
            val_points_count = len(full_dataset) - train_points_count
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
            )

        if stage == 'test' or stage is None:
            test_dataset_path = self.prepared_data_path / "test_dataset.hdf5"
            self.test_dataset = self.dataset_class(test_dataset_path,
                                                   transforms=self.transforms,
                                                   inverse_transforms=self.inverse_transforms,
                                                   include_distances_to_melting_pool=self.include_distances_to_melting_pool,
                                                   include_coordinates=self.input_channels[0].return_coordinates)
