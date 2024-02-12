from os import listdir
from os.path import isfile, join
from pathlib import Path
import re
from typing import List, Dict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from leap3d.datasets import ExtraParam, Channel
from leap3d.scanning import ScanParameters, ScanResults


def prepare_raw_data(scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                     raw_data_dir: str | Path, prepared_data_path: str| Path, cases: int | List=None,
                     channel=Channel.TEMPERATURE, extra_params=[], is_3d=False):
    def write_to_datasets(train_dataset, extra_train_dataset, target_dataset, training_points, extra_train_points, targets, offset):
        number_of_points = len(training_points)
        total_number_of_points = number_of_points + offset
        resize_datasets(train_dataset, extra_train_dataset, target_dataset, total_number_of_points)

        train_dataset[offset:] = np.array(training_points)
        train_dataset.flush()
        if extra_params is not None:
            extra_train_dataset[offset:] = np.array(extra_train_points)
            extra_train_dataset.flush()
        target_dataset[offset:] = np.array(targets)
        target_dataset.flush()

        return total_number_of_points
    def resize_datasets(train_dataset, extra_train_dataset, target_dataset, number_of_points: int):
        if train_dataset.len() < number_of_points:
            train_dataset.resize(number_of_points, axis=0)
            if extra_train_dataset is not None:
                extra_train_dataset.resize(number_of_points, axis=0)
            target_dataset.resize(number_of_points, axis=0)

    # TODO: Refactor
    params_file = np.load(scan_parameters_filepath)
    rough_coordinates = np.load(rough_coordinates_filepath)

    offset = 0
    with h5py.File(prepared_data_path, 'w') as f:
        train_dset_shape = (1, 64, 64, 64) if is_3d else (1, 64, 64)
        train_dset_maxshape = (None, 64, 64, 64) if is_3d else (None, 64, 64)
        train_dset = f.create_dataset("input", train_dset_shape, maxshape=train_dset_maxshape, chunks=True)
        if extra_params is not None:
            extra_train_dset = f.create_dataset("train_extra_params", (1, len(extra_params)), maxshape=(None, len(extra_params)), chunks=True)
        target_dset_shape = (1, 64, 64, 64) if is_3d else (1, 64, 64)
        target_dset_maxshape = (None, 64, 64, 64) if is_3d else (None, 64, 64)
        target_dset = f.create_dataset("target", target_dset_shape, maxshape=target_dset_maxshape, chunks=True)
        scan_result_filepaths = get_scan_result_case_ids_and_filepaths(raw_data_dir, cases)
        for case_id, scan_result_filepath in scan_result_filepaths:
            scan_parameters = ScanParameters(params_file, rough_coordinates, case_id)

            data_generator = prepare_scan_results(scan_result_filepath, scan_parameters,
                                                  channel=channel, extra_params=extra_params, case_id=case_id, is_3d=is_3d)
            train_buffer, extra_train_buffer, target_buffer = [], [], []
            for training_points, extra_train_points, targets in data_generator:
                train_buffer.append(training_points)
                extra_train_buffer.append(extra_train_points)
                target_buffer.append(targets)
                if len(train_buffer) == 1000:
                    write_to_datasets(train_dset, extra_train_dset if extra_params is not None else None, target_dset, train_buffer, extra_train_buffer, target_buffer, offset)
                    offset += len(train_buffer)
                    train_buffer, extra_train_buffer, target_buffer = [], [], []

            if len(train_buffer) != 0:
                write_to_datasets(train_dset, extra_train_dset if extra_params is not None else None, target_dset, train_buffer, extra_train_buffer, target_buffer, offset)
                offset += len(train_buffer)
                train_buffer, extra_train_buffer, target_buffer = [], [], []


def prepare_scan_results(scan_result_filepath, scan_parameters, channel=Channel.TEMPERATURE_AROUND_LASER, extra_params=[], case_id=None, is_3d=False):
    def get_temperature_grid_around_laser_channel(
            timestep,
            scan_results,
            scan_parameters,
            include_high_resolution_points,
            is_3d):
        return scan_results.get_interpolated_grid_around_laser(timestep, 16, 0.25, scan_parameters, include_high_resolution_points, is_3d)[1]

    scan_results = ScanResults(scan_result_filepath)
    print(case_id)
    for i in range(0, scan_results.total_timesteps):
        extra_params_channels = []

        if channel == Channel.TEMPERATURE_AROUND_LASER:
            training_point = get_temperature_grid_around_laser_channel(i, scan_results, scan_parameters, False, is_3d)
        else:
            raise ValueError(f"Channel {channel} not supported")

        if extra_params is not None:
            for extra_param in extra_params:
                if extra_param == ExtraParam.SCANNING_ANGLE:
                    extra_params_channels.append(scan_parameters.scanning_angle)
                if extra_param == ExtraParam.LASER_POWER:
                    extra_params_channels.append(scan_results.get_laser_power_at_timestep(i))
                if extra_param == ExtraParam.LASER_RADIUS:
                    extra_params_channels.append(scan_results.get_laser_radius_at_timestep(i))

        target_point = get_temperature_grid_around_laser_channel(i, scan_results, scan_parameters, True, is_3d)

        yield np.array(training_point), np.array(extra_params_channels), np.array(target_point)


def get_scan_result_case_ids_and_filepaths(data_dir: str="path/to/dir", cases: int | List=None):
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



class UnetInterpolationDataset(Dataset):
    def __init__(self, data_dir,
                 train: bool=True,
                 extra_params: List[ExtraParam]=None,
                 transforms: Dict[str, callable]={}, inverse_transforms: Dict[str, callable]={}):
        super().__init__()

        self.train = train

        self.data_filepath = Path(data_dir) / (f"{'test_' if not train else ''}unet_interpolation_dataset.hdf5")
        self.data_file = h5py.File(self.data_filepath, 'r')

        self.inputs = self.data_file['input']
        self.targets = self.data_file['target']
        self.extra_params = extra_params
        if self.extra_params is not None:
            self.train_extra_params = self.data_file['train_extra_params']

        default_transform = lambda x: x

        self.input_transform = transforms.get('input', default_transform)
        self.extra_params_transform = transforms.get('extra_params', default_transform)
        self.target_transform = transforms.get('target', default_transform)

        self.inverse_input_transform = inverse_transforms.get('input', default_transform)
        self.inverse_target_transform = inverse_transforms.get('target', default_transform)

    def __len__(self):
        return self.inputs.len()

    def __getitem__(self, idx):
        data = self.inputs[idx]
        extra_params = self.train_extra_params[idx] if self.extra_params is not None else torch.empty((data.shape[0], 0))
        target = self.targets[idx]

        data = self.input_transform(data)
        extra_params = self.extra_params_transform(extra_params)
        target = self.target_transform(target)

        return data, extra_params, target

    def __del__(self):
        self.data_file.close()


class UnetInterpolationDataModule(pl.LightningDataModule):
    def __init__(self, scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_dir: str | Path, processed_data_dir: str | Path,
                 is_3d: bool=False, batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 extra_params: List[ExtraParam]=None,
                 transforms: Dict[str, callable]={}, inverse_transforms: Dict[str, callable]={},
                 force_prepare=False, num_workers=0):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.scan_parameters_filepath = scan_parameters_filepath
        self.rough_coordinates_filepath = rough_coordinates_filepath
        self.raw_data_dir = Path(raw_data_dir) if isinstance(raw_data_dir, str) else raw_data_dir
        self.processed_data_dir = Path(processed_data_dir) if isinstance(processed_data_dir, str) else processed_data_dir

        self.is_3d = is_3d
        self.batch_size = batch_size

        self.train_cases = train_cases
        self.test_cases = test_cases

        self.extra_params = extra_params
        self.transforms = transforms
        self.inverse_transforms = inverse_transforms

        self.force_prepare = force_prepare
        self.num_workers = num_workers

    def prepare_data(self):
        prepared_train_filepath = self.processed_data_dir / "unet_interpolation_dataset.hdf5"
        prepared_test_filepath = self.processed_data_dir / "test_unet_interpolation_dataset.hdf5"

        if self.force_prepare or not isfile(prepared_train_filepath):
            prepare_raw_data(
                self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_train_filepath,
                self.train_cases, Channel.TEMPERATURE_AROUND_LASER, self.extra_params, is_3d=self.is_3d)
        if self.force_prepare or not isfile(prepared_test_filepath):
            prepare_raw_data(
                self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_test_filepath,
                self.test_cases, Channel.TEMPERATURE_AROUND_LASER, self.extra_params, is_3d=self.is_3d)

    def setup(self, stage=None, split_ratio=0.8):
        self.test_dataset = UnetInterpolationDataset(self.processed_data_dir, train=False,
                                                  extra_params=self.extra_params,
                                                  transforms=self.transforms, inverse_transforms=self.inverse_transforms)
        full_dataset = UnetInterpolationDataset(self.processed_data_dir, train=True,
                                                  extra_params=self.extra_params,
                                                  transforms=self.transforms, inverse_transforms=self.inverse_transforms)

        train_points_count = int(np.ceil(len(full_dataset) * split_ratio))
        val_points_count = len(full_dataset) - train_points_count
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def teardown(self, stage: str) -> None:
        pass