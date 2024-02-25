from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
import torch
from torch.utils.data import random_split
from leap3d.contour import get_melting_pool_contour_2d

from leap3d.datasets import LEAPDataModule, LEAPDataset
from leap3d.datasets.channels import RoughTemperatureAroundLaser, TemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius
import shapely.geometry as sg


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
        self.target_distances_to_melting_pool = self.data_file['distances_to_melting_pool']

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
        target_distances_to_melting_pool = self.target_distances_to_melting_pool[idx]

        melting_pool_idx_start, melting_pool_idx_end = self.melting_pool_indices[idx]
        melting_pool = self.melting_pool[int(melting_pool_idx_start):int(melting_pool_idx_end)]
        laser_data = self.laser_data[idx]

        melting_pool = self.melting_pool_transform(melting_pool)
        laser_data = self.laser_data_transform(laser_data)

        return input, extra_input, target, target_coordinates, melting_pool, laser_data, target_distances_to_melting_pool


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
        self.dataset_class = MLPInterpolationDataset
        self.melting_pool_offset = 0
        self.melt_pool_shape = (9,)
        self.rough_coordinates_box_size = 32
        self.interpolated_coordinates_scale = 0.25

    def create_h5_datasets(self, h5py_file):
        datasets = super().create_h5_datasets(h5py_file)
        datasets['target_coordinates'] = self.create_h5_target_coordinates_dataset(h5py_file)
        datasets['melting_pool'] = self.create_h5_melting_pool_dataset(h5py_file)
        datasets['melting_pool_indices'] = self.create_h5_melting_pool_indices_dataset(h5py_file)
        datasets['laser_data'] = self.create_h5_laser_data_dataset(h5py_file)
        datasets['distances_to_melting_pool'] = self.create_h5_distances_to_melting_pool_dataset(h5py_file)

        return datasets

    def create_h5_target_coordinates_dataset(self, h5py_file):
        dset_shape = [1, 6 if self.is_3d else 4]  # x_min, x_max, y_min, y_max, [z_min, z_max]
        dset_maxshape = [None, *dset_shape[1:]]

        return h5py_file.create_dataset("target_coordinates", dset_shape, maxshape=dset_maxshape, chunks=True)

    def create_h5_melting_pool_dataset(self, h5py_file):
        dset_shape = (1, *self.melt_pool_shape)  # x, y, z, temperature, temperature gradient x, temperature gradient y, temperature gradient z, phase indicator, distance_to_contour
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

    def create_h5_distances_to_melting_pool_dataset(self, h5py_file):
        dset_shape = (1, *self.target_shape)
        dset_maxshape = (None, *dset_shape[1:])

        return h5py_file.create_dataset("distances_to_melting_pool", dset_shape, maxshape=dset_maxshape, chunks=True)

    def get_data_generator_timestep(self, scan_results, scan_parameters, timestep):
        data = super().get_data_generator_timestep(scan_results, scan_parameters, timestep)
        data['target_coordinates'] = self.get_target_coordinates(scan_results, scan_parameters, timestep)

        contour_shape = self.get_melting_pool_contour(scan_results, scan_parameters, timestep)

        melting_pool, melting_pool_indices = self.get_melting_pool(scan_results, scan_parameters, contour_shape, timestep)
        data['melting_pool'] = melting_pool
        data['melting_pool_indices'] = melting_pool_indices
        data['laser_data'] = self.get_laser_data(scan_results, timestep)

        data['distances_to_melting_pool'] = self.get_target_distances_to_melting_pool_2d(scan_results, scan_parameters, contour_shape, timestep)

        return data

    def get_target_coordinates(self, scan_results, scan_parameters, timestep):
        laser_x, laser_y = scan_results.laser_data[timestep][:2]
        bounds = scan_parameters.get_bounds_around_position(laser_x, laser_y, self.rough_coordinates_box_size)
        if self.is_3d:
            return np.array(bounds)
        return np.array(bounds[:4])

    def get_melting_pool_contour(self, scan_results, scan_parameters, timestep):
        temperature_high_res = scan_results.get_melt_pool_temperature_grid(scan_parameters, timestep)
        if self.is_3d:
            raise NotImplementedError("3D not implemented yet")
        else:
            contours = get_melting_pool_contour_2d(temperature_high_res, top_k=1)[0]
        if len(contours) == 0:
            laser_position = scan_results.get_laser_coordinates_at_timestep(timestep)
            return sg.Point(laser_position)

        contour = np.array(contours[0])
        # Rescale coordinates from unit distance into meters
        contour = ((contour - 128) / 128) * 0.0012
        return sg.Polygon(contour)


    def get_melting_pool(self, scan_results, scan_parameters, contour_shape, timestep):
        melt_pool_data = scan_results.melt_pool[timestep]
        if melt_pool_data == [[]]:
            melt_pool_data = np.zeros((0, *self.melt_pool_shape))
            melting_pool_indices = np.array([self.melting_pool_offset, self.melting_pool_offset])
            return melt_pool_data, melting_pool_indices

        melt_pool_data = np.array(melt_pool_data)

        # filter out points that are not in the last layer if 2D
        if not self.is_3d:
            max_z = np.max(melt_pool_data[:, 2])
            melt_pool_data = melt_pool_data[melt_pool_data[:, 2] == max_z]

        if isinstance(contour_shape, sg.Point):
            distances = [contour_shape.distance(sg.Point(p[:2])) for p in melt_pool_data[:, :2]]
        else:
            distances = [np.abs(contour_shape.exterior.distance(sg.Point(p[:2]))) for p in melt_pool_data[:, :2]]
        melt_pool_data = np.column_stack((melt_pool_data, distances))

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

    def get_target_distances_to_melting_pool_2d(self, scan_results, scan_parameters, contour_shape, timestep):
        laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
        points = scan_parameters.get_coordinates_around_position(laser_x, laser_y,
                                                                 self.rough_coordinates_box_size,
                                                                 scale=self.interpolated_coordinates_scale,
                                                                 is_3d=self.is_3d)
        if isinstance(contour_shape, sg.Point):
            distances = [contour_shape.distance(sg.Point(p[:2])) for p in points]
        else:
            distances = [np.abs(contour_shape.exterior.distance(sg.Point(p[:2]))) for p in points]

        return np.array(distances).reshape(self.target_shape)
