from os import listdir
from os.path import isfile, join
from pathlib import Path
import re
from typing import Callable, Dict, List

import numpy as np
import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from leap3d.datasets.channels import LaserPosition, RoughTemperature, RoughTemperatureDifference, ScanningAngle, LaserPower, LaserRadius
from leap3d.scanning import ScanParameters, ScanResults


class LEAPDataset(Dataset):
    def __init__(self, data_filepath: str="path/to/dir",
                 transforms: Dict={}, inverse_transforms: Dict={}):
        super().__init__()

        self.data_filepath = Path(data_filepath)
        self.data_file = h5py.File(self.data_filepath, 'r')
        self.inputs = self.data_file['input']
        self.extra_inputs = self.data_file['extra_input']
        self.targets = self.data_file['target']

        default_transform = lambda x: x
        self.input_transform = transforms.get('input', default_transform)
        self.extra_input_transform = transforms.get('extra_input', default_transform)
        self.target_transform = transforms.get('target', default_transform)

        self.input_inverse_transform = inverse_transforms.get('input', default_transform)
        self.extra_input_inverse_transform = inverse_transforms.get('extra_input', default_transform)
        self.target_inverse_transform = inverse_transforms.get('target', default_transform)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        extra_input = self.extra_inputs[idx]
        target = self.targets[idx]

        input = self.input_transform(input)
        extra_input = self.extra_input_transform(extra_input)
        target = self.target_transform(target)

        return input, extra_input, target

    def unnormalize_temperature(self, temperature):
        return self.input_inverse_transform(temperature)

    def unnormalize_target(self, target):
        return self.target_inverse_transform(target)

    def __del__(self):
        self.data_file.close()


class LEAPDataModule(pl.LightningDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_directory: str="path/to/dir", prepared_data_path: str="path/to/file",
                 is_3d: bool=False,
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 input_shape: List[int]=[64, 64, 16], target_shape: List[int]=[64, 64, 16],
                 input_channels=[LaserPosition, RoughTemperature], extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
                 target_channels=[RoughTemperatureDifference],
                 transforms: Dict[str, Callable]={}, inverse_transforms: Dict[str, Callable]={},
                 force_prepare=False,
                 num_workers=0):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.scan_parameters_filepath = Path(scan_parameters_filepath)
        self.rough_coordinates_filepath = Path(rough_coordinates_filepath)
        self.raw_data_directory = Path(raw_data_directory)
        self.prepared_data_path = Path(prepared_data_path)

        self.is_3d = is_3d
        self.batch_size = batch_size
        self.train_cases = train_cases
        self.test_cases = test_cases

        self.input_shape = input_shape if is_3d else input_shape[:2]
        self.target_shape = target_shape if is_3d else target_shape[:2]

        self.input_channels = [channel(is_3d=self.is_3d) for channel in input_channels]
        self.input_channel_len = sum([channel.channels for channel in self.input_channels])
        self.extra_input_channels = [channel(is_3d=self.is_3d) for channel in extra_input_channels]
        self.extra_input_channel_len = sum([channel.channels for channel in self.extra_input_channels])
        self.target_channels = [channel(is_3d=self.is_3d) for channel in target_channels]
        self.target_channel_len = sum([channel.channels for channel in self.target_channels])

        self.transforms = transforms
        self.inverse_transforms = inverse_transforms

        self.force_prepare = force_prepare
        self.num_workers = num_workers

    def prepare_data(self):
        self.prepare_train_data()
        self.prepare_test_data()

    def check_if_prepared_data_exists(self, path):
        return path.exists()

    def prepare_train_data(self):
        processed_data_filepath = self.prepared_data_path / "dataset.hdf5"

        if not self.force_prepare and self.check_if_prepared_data_exists(processed_data_filepath):
            return

        self.process_data(self.train_cases, processed_data_filepath)

    def prepare_test_data(self):
        processed_data_filepath = self.prepared_data_path / "test_dataset.hdf5"

        if not self.force_prepare and self.check_if_prepared_data_exists(processed_data_filepath):
            return

        self.process_data(self.test_cases, processed_data_filepath)

    def process_data(self, cases, processed_data_filepath):
        scan_case_id_and_result_filepaths = self.get_scan_case_ids_and_result_filepaths(cases)
        params_file = np.load(self.scan_parameters_filepath)
        rough_coordinates = np.load(self.rough_coordinates_filepath)

        offset = 0
        with h5py.File(processed_data_filepath, 'w') as f:
            input_dset = self.create_h5_input_dataset(f)
            extra_input_dset = self.create_h5_extra_input_dataset(f)
            target_dset = self.create_h5_target_dataset(f)

            for case_id, scan_result_filepath in scan_case_id_and_result_filepaths:
                data_generator = self.get_data_generator(case_id, scan_result_filepath, params_file, rough_coordinates)
                offset += self.write_data_to_h5(data_generator, input_dset, extra_input_dset, target_dset, offset)

    def create_h5_input_dataset(self, h5py_file):
        input_dset_shape = [1, self.input_channel_len, *self.input_shape]
        input_dset_maxshape = [None, self.input_channel_len, *self.input_shape]

        return h5py_file.create_dataset("input", input_dset_shape, maxshape=input_dset_maxshape, chunks=True)

    def create_h5_extra_input_dataset(self, h5py_file):
        extra_input_dset_shape = [1, self.extra_input_channel_len]
        extra_input_dset_maxshape = [None, self.extra_input_channel_len]

        return h5py_file.create_dataset("extra_input", extra_input_dset_shape, maxshape=extra_input_dset_maxshape, chunks=True)

    def create_h5_target_dataset(self, h5py_file):
        target_dset_shape = (1, self.target_channel_len, *self.target_shape)
        target_dset_maxshape = (None, self.target_channel_len, *self.target_shape)

        return h5py_file.create_dataset("target", target_dset_shape, maxshape=target_dset_maxshape, chunks=True)

    def get_data_generator(self, case_id, scan_result_filepath, params_file, rough_coordinates):
        scan_results = ScanResults(scan_result_filepath)
        scan_parameters = ScanParameters(params_file, rough_coordinates, case_id)

        for timestep in range(scan_results.total_timesteps - 1):
            input_channels = []
            extra_input_channels = []
            target_channels = []

            for channel in self.input_channels:
                input_channels.extend(
                    channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=timestep))

            for channel in self.extra_input_channels:
                extra_input_channels.extend(channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=timestep))

            for channel in self.target_channels:
                target_channels.extend(channel.get(scan_parameters=scan_parameters, scan_results=scan_results, timestep=timestep))

            yield input_channels, extra_input_channels, target_channels

    def write_to_h5_dataset(self, dset, data, offset):
        dset.resize(offset + len(data), axis=0)
        dset[offset:] = np.array(data)
        dset.flush()

    def write_data_to_h5(self, data_generator, input_dset, extra_input_dset, target_dset, offset):
        train_buffer, extra_train_buffer, target_buffer = [], [], []
        points_written = 0
        buffer_size = 2000
        for input_points, extra_input_points, target_points in data_generator:
            train_buffer.append(input_points)
            extra_train_buffer.append(extra_input_points)
            target_buffer.append(target_points)
            if len(train_buffer) == buffer_size:
                self.write_to_h5_dataset(input_dset, train_buffer, offset)
                self.write_to_h5_dataset(extra_input_dset, extra_train_buffer, offset)
                self.write_to_h5_dataset(target_dset, target_buffer, offset)
                offset += buffer_size
                points_written += buffer_size
                train_buffer, extra_train_buffer, target_buffer = [], [], []

        if len(train_buffer) > 0:
            self.write_to_h5_dataset(input_dset, train_buffer, offset)
            self.write_to_h5_dataset(extra_input_dset, extra_train_buffer, offset)
            self.write_to_h5_dataset(target_dset, target_buffer, offset)
            points_written += len(train_buffer)

        return points_written

    def get_scan_case_ids_and_result_filepaths(self, data_dir: str="path/to/dir", cases: int | List=None):
        if type(cases) == int:
            cases = range(cases)

        filenames = [
            filename for filename in listdir(data_dir)
                if isfile(join(data_dir, filename)) and re.match(r"^case_\d*\.npz$", filename) is not None]
        print(filenames, re.search(r'(\d+)', filenames[0]).group(0))
        if cases is not None:
            filenames = filter(lambda filename: int(re.search(r'(\d+)', filename).group(0) or -1) in cases, filenames)

        # Return case_id and filename
        return [(int(re.search(r'(\d+)', filename).group(0)), Path(data_dir) / filename) for filename in filenames]

    def setup(self, stage: str, split_ratio: float=0.8):
        if stage == 'fit' or stage is None:
            full_dataset_path = self.prepared_data_path / "dataset.hdf5"
            full_dataset = LEAPDataset(full_dataset_path, transforms=self.transforms, inverse_transforms=self.inverse_transforms)

            train_points_count = int(np.ceil(len(full_dataset) * split_ratio))
            val_points_count = len(full_dataset) - train_points_count
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
            )

        if stage == 'test' or stage is None:
            test_dataset_path = self.prepared_data_path / "test_dataset.hdf5"
            self.test_dataset = LEAPDataset(test_dataset_path, transforms=self.transforms, inverse_transforms=self.inverse_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        pass
