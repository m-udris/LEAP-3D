from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from leap3d.datasets import LEAPDataModule, LEAPDataset
from leap3d.datasets.channels import LowResRoughTemperatureAroundLaser, MeltPoolPointChunk, ScanningAngle, LaserPower, LaserRadius


class MLPInterpolationChunkDataset(LEAPDataset):
    def __init__(self, data_filepath: str="path/to/dir",
                 transforms: Dict={}, inverse_transforms: Dict={},
                 is_3d: bool=False,
                 include_melting_pool: bool=False,
                 include_distances_to_melting_pool: bool=False,
                 include_input_coordinates: bool=False):
        super().__init__(
            data_filepath=data_filepath,
            transforms=transforms,
            inverse_transforms=inverse_transforms
        )
        self.include_melting_pool = include_melting_pool
        self.include_distances_to_melting_pool = include_distances_to_melting_pool
        self.include_input_coordinates = include_input_coordinates
        self.is_3d = is_3d

    def __getitem__(self, idx):
        input = self.inputs[idx]
        if not self.include_input_coordinates:
            input = input[..., -1:, :, :, :] if self.is_3d else input[..., -1:, :, :]
        extra_input = self.extra_inputs[idx]
        target = self.targets[idx]

        input = self.input_transform(input)
        extra_input = self.extra_input_transform(extra_input)
        target = self.target_transform(target)

        return input, extra_input, target


class MLPInterpolationChunkDataModule(LEAPDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_directory: str="path/to/dir", prepared_data_path: str="path/to/file",
                 is_3d: bool=False,
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 input_shape: List[int]=[64, 64, 16], target_shape: List[int]=[64,64,16],
                 input_channels=[LowResRoughTemperatureAroundLaser], extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
                 target_channels=[MeltPoolPointChunk],
                 transforms: Dict[str, Callable]={}, inverse_transforms: Dict[str, Callable]={},
                 include_distances_to_melting_pool: bool=False,
                 offset_ratio: float=None,
                 force_prepare=False,
                 is_preshuffled=False,
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
        self.dataset_class = MLPInterpolationChunkDataset
        self.melting_pool_offset = 0
        self.melt_pool_shape = (9,) if include_distances_to_melting_pool else (8,)
        self.interpolated_coordinates_offset_ratio = offset_ratio
        self.include_distances_to_melting_pool = include_distances_to_melting_pool
        self.is_preshuffled = is_preshuffled

    def setup(self, stage: str, split_ratio: float=0.8):
        if stage == 'fit' or stage is None:
            full_dataset_path = self.prepared_data_path / "dataset.hdf5"
            full_dataset = self.dataset_class(
                full_dataset_path,
                is_3d=self.is_3d,
                transforms=self.transforms,
                inverse_transforms=self.inverse_transforms,
                include_distances_to_melting_pool=self.include_distances_to_melting_pool,
                include_input_coordinates=self.input_channels[0].return_coordinates)

            train_points_count = int(np.ceil(len(full_dataset) * split_ratio))
            val_points_count = len(full_dataset) - train_points_count
            if self.is_preshuffled:
                self.train_dataset = torch.utils.data.Subset(full_dataset, range(train_points_count))
                self.val_dataset = torch.utils.data.Subset(full_dataset, range(train_points_count, train_points_count + val_points_count))
            else:
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
                )

        if stage == 'test' or stage is None:
            test_dataset_path = self.prepared_data_path / "test_dataset.hdf5"
            self.test_dataset = self.dataset_class(test_dataset_path,
                                                   is_3d=self.is_3d,
                                                   transforms=self.transforms,
                                                   inverse_transforms=self.inverse_transforms,
                                                   include_distances_to_melting_pool=self.include_distances_to_melting_pool,
                                                   include_input_coordinates=self.input_channels[0].return_coordinates)

    def write_data_to_h5(self, data_generator, datasets, offset):
        buffers = {name: [] for name in datasets.keys()}
        points_written = 0
        buffer_size = 200

        items_in_buffer = 0
        for points in data_generator:
            target_point_chunks = points.pop('target')

            for data_name, values in points.items():
                for target_point_chunk in target_point_chunks:
                    buffers[data_name].append(values)
            if len(target_point_chunks) > 1:
                print(f'Target Chunks: {len(target_point_chunks)}')
            for target_point_chunk in target_point_chunks:
                buffers['target'].append(target_point_chunk)
                items_in_buffer += 1

            if items_in_buffer >= buffer_size:
                for dset_name, dset in datasets.items():
                    self.write_to_h5_dataset(dset, buffers[dset_name], offset)
                offset += items_in_buffer
                points_written += items_in_buffer
                buffers = {name: [] for name in datasets.keys()}
                items_in_buffer = 0

        if items_in_buffer > 0:
            for dset_name, dset in datasets.items():
                self.write_to_h5_dataset(dset, buffers[dset_name], offset)
            points_written += items_in_buffer

        return points_written

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, shuffle=not self.is_preshuffled)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not self.is_preshuffled)

    def test_dataloader(self, **kwargs):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not self.is_preshuffled, **kwargs)