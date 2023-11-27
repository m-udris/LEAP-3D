from enum import Enum
import logging
from os import listdir
from os.path import isfile, join
from pathlib import Path
import re
from typing import List

import numpy as np
import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from leap3d.config import DATA_DIR, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SAFETY_OFFSET, MELTING_POINT, TIMESTEP_DURATION
from leap3d.scanning import ScanParameters, ScanResults
from leap3d.laser import convert_laser_position_to_grid


class Channel(Enum):
    LASER_POSITION = 1
    TEMPERATURE = 2


class ExtraParam(Enum):
    SCANNING_ANGLE = 1
    LASER_POWER = 2
    LASER_RADIUS = 3


def check_if_prepared_data_exists(filepath: str| Path):
    # Might be a good idea to add additional checks, e.g. whether
    # dataset metadata matches
    return isfile(filepath)


def prepare_raw_data(scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                     raw_data_dir: str | Path, prepared_data_path: str| Path, cases: int | List=None,
                     window_size: int=1, window_step_size: int=1, channels=[Channel.LASER_POSITION, Channel.TEMPERATURE], extra_params=[], is_3d=False):
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
        train_dset_shape = (1, window_size, len(channels)+1, 64, 64, 16) if is_3d else (1, window_size, len(channels)+1, 64, 64)
        train_dset_maxshape = (None, window_size, len(channels)+1, 64, 64, 16) if is_3d else (None, window_size, len(channels)+1, 64, 64)
        train_dset = f.create_dataset("train", train_dset_shape, maxshape=train_dset_maxshape, chunks=True)
        if extra_params is not None:
            extra_train_dset = f.create_dataset("train_extra_params", (1, window_size, len(extra_params)), maxshape=(None, window_size, len(extra_params)), chunks=True)
        target_dset_shape = (1, window_size, 1, 64, 64, 16) if is_3d else (1, window_size, 1, 64, 64)
        target_dset_maxshape = (None, window_size, 1, 64, 64, 16) if is_3d else (None, window_size, 1, 64, 64)
        target_dset = f.create_dataset("target", target_dset_shape, maxshape=target_dset_maxshape, chunks=True)
        scan_result_filepaths = get_scan_result_case_ids_and_filepaths(raw_data_dir, cases)
        for case_id, scan_result_filepath in scan_result_filepaths:
            scan_parameters = ScanParameters(params_file, rough_coordinates, case_id)

            data_generator = prepare_scan_results(scan_result_filepath, scan_parameters,
                                                                          window_size=window_size, window_step_size=window_step_size, channels=channels, extra_params=extra_params, case_id=case_id, is_3d=is_3d)
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


def prepare_scan_results(scan_result_filepath, scan_parameters, window_size=1, window_step_size=1, channels=[Channel.LASER_POSITION, Channel.TEMPERATURE], extra_params=[],case_id=None, is_3d=False):
    # TODO: Refactor into enum?
    def get_temperature_channel(timestep, scan_results, is_3d):
        if is_3d:
            return scan_results.rough_temperature_field[timestep]

        return scan_results.rough_temperature_field[timestep, :, :, -1]

    def get_temperature_diff_channel(timestep, scan_results, is_3d):
        temperature = scan_results.rough_temperature_field[timestep]
        next_step_temperature = scan_results.rough_temperature_field[timestep + 1]

        next_step_diffferences = next_step_temperature - temperature

        if is_3d:
            return next_step_diffferences

        return next_step_diffferences[:, :, -1]

    def get_laser_position_channels(timestep, scan_results, scan_parameters, is_3d):
        laser_data = scan_results.get_laser_data_at_timestep(timestep)
        next_laser_data = scan_results.get_laser_data_at_timestep(timestep + 1)

        laser_position_grid = convert_laser_position_to_grid(laser_data[0], laser_data[1], scan_parameters.rough_coordinates, is_3d=is_3d)
        next_laser_position_grid = convert_laser_position_to_grid(next_laser_data[0], next_laser_data[1], scan_parameters.rough_coordinates, is_3d=is_3d)

        return [laser_position_grid, next_laser_position_grid]

    scan_results = ScanResults(scan_result_filepath)

    training_points = []
    extra_params_points = []
    targets = []

    for timestep in range(0, scan_results.total_timesteps - window_size, window_step_size):
        training_points_window = []
        extra_params_window = []
        targets_window = []

        for i in range(timestep, timestep + window_size):
            training_point_channels = []
            extra_params_channels = []
            targets_channels = []

            for channel in channels:
                if channel == Channel.LASER_POSITION:
                    training_point_channels.extend(get_laser_position_channels(i, scan_results, scan_parameters, is_3d))
                if channel == Channel.TEMPERATURE:
                    training_point_channels.append(get_temperature_channel(i, scan_results, is_3d))

            if extra_params is not None:
                for extra_param in extra_params:
                    if extra_param == ExtraParam.SCANNING_ANGLE:
                        extra_params_channels.append(scan_parameters.scanning_angle)
                    if extra_param == ExtraParam.LASER_POWER:
                        extra_params_channels.append(scan_results.get_laser_power_at_timestep(i))
                    if extra_param == ExtraParam.LASER_RADIUS:
                        extra_params_channels.append(scan_results.get_laser_radius_at_timestep(i))

            temperature_diff_channel = get_temperature_diff_channel(i, scan_results, is_3d)
            # if np.max(temperature_diff_channel) > 1000:
            #     logging.warning(f"Temperature difference above 1000 at timestep {timestep} for case {case_id}")
            targets_channels.append(temperature_diff_channel)

            training_points_window.append(np.array(training_point_channels))
            extra_params_window.append(np.array(extra_params_channels))
            targets_window.append(targets_channels)

        yield np.array(training_points_window), np.array(extra_params_window), np.array(targets_window)


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


class LEAP3DDataset(Dataset):
    def __init__(self, data_dir: str="path/to/dir",
                 train: bool=True, extra_params: List[ExtraParam]=None,
                 transform=None, target_transform=None, extra_params_tranform=None, is_eval=False):
        super().__init__()
        self.train = train
        self.is_eval = is_eval
        self.data_filepath = Path(data_dir) / ("eval_dataset.hdf5" if is_eval else "dataset.hdf5" if train else "test_dataset.hdf5")
        self.data_file = h5py.File(self.data_filepath, 'r')
        self.x_train = self.data_file['train']
        self.extra_params = extra_params
        if self.extra_params is not None:
            self.train_extra_params = self.data_file['train_extra_params']
        self.targets = self.data_file['target']
        self.transform = transform
        self.extra_params_tranform = extra_params_tranform
        self.target_transform = target_transform
        self.temperature_mean = None
        self.temperature_variance = None
        self.target_mean = None
        self.target_variance = None

    def __len__(self):
        return self.x_train.len()

    def __getitem__(self, idx):
        data = self.x_train[idx]
        extra_params = self.train_extra_params[idx] if self.extra_params is not None else torch.empty((data.shape[0], 0))
        target = self.targets[idx]

        if self.transform is not None:
            data = self.transform(data)
        if self.extra_params_tranform is not None:
            extra_params = self.extra_params_tranform(extra_params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, extra_params, target

    def get_temperature_mean_and_variance(self, print_progress=False):
        if all([self.temperature_mean, self.temperature_variance, self.target_mean, self.target_variance]):
            return self.temperature_mean, self.temperature_variance, self.target_mean, self.target_variance
        # Cast to float64 to avoid negative variance due to floating point precision errors
        temperature_mean = np.zeros_like(self.x_train[0, :, -1]).astype(np.float64)
        temperature_squares = np.zeros_like(self.x_train[0, :, -1]).astype(np.float64)

        target_mean = np.zeros_like(self.targets[0, :, 0]).astype(np.float64)
        target_squares = np.zeros_like(self.targets[0, :, 0]).astype(np.float64)

        for index, (x_train, extra_params, y_target) in enumerate(self):
            if print_progress and index % 100 == 0:
                print(index, end='\r')
            temperature_mean += x_train[:,-1]
            temperature_squares += np.square(x_train[:,-1])
            target_mean += y_target[:,0]
            target_squares += np.square(y_target[:,0])

        len_self = len(self)
        temperature_mean = temperature_mean.mean() / len_self
        temperature_squares = temperature_squares.mean() / len_self
        temperature_variance = np.sqrt(temperature_squares - temperature_mean**2)

        target_mean = target_mean.mean() / len_self
        target_squares = target_squares.mean() / len_self
        target_variance = np.sqrt(target_squares - target_mean**2)

        self.temperature_mean = temperature_mean
        self.temperature_variance = temperature_variance
        self.target_mean = target_mean
        self.target_variance = target_variance

        return temperature_mean, temperature_variance, target_mean, target_variance

    def get_temperature_z_scores(self):
        temperature_mean, temperature_variance, target_mean, target_variance = self.get_temperature_mean_and_variance()

        x_train_reshaped = self.x_train[:, :, -1].reshape(-1)
        if self.transform is not None:
            x_train_reshaped = self.transform(x_train_reshaped)
        targets_reshaped = self.targets[:, :, 0].reshape(-1)
        if self.target_transform is not None:
            targets_reshaped = self.target_transform(targets_reshaped)

        return (x_train_reshaped - temperature_mean) / temperature_variance, (targets_reshaped - target_mean) / target_variance

    def __del__(self):
        self.data_file.close()


class LEAP3DDataModule(pl.LightningDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_dir: str="path/to/dir", prepared_data_path: str="path/to/dir",
                 is_3d: bool=False,
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None, eval_cases: int | List=None,
                 window_size: int=1, window_step_size: int=1,
                 channels=[Channel.LASER_POSITION, Channel.TEMPERATURE], extra_params=[ExtraParam.SCANNING_ANGLE, ExtraParam.LASER_POWER, ExtraParam.LASER_RADIUS],
                 transform: callable=None, target_transform: callable=None, extra_params_transform: callable=None,
                 force_prepare=False,
                 num_workers=1):
        super().__init__()
        self.scan_parameters_filepath = scan_parameters_filepath
        self.rough_coordinates_filepath = rough_coordinates_filepath
        self.raw_data_dir = raw_data_dir
        self.prepared_data_path = prepared_data_path
        self.batch_size = batch_size
        self.train_cases = train_cases
        self.test_cases = test_cases
        self.eval_cases = eval_cases
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.force_prepare = force_prepare
        self.num_workers = num_workers
        self.channels = channels
        self.extra_params = extra_params
        self.tranform = transform
        self.extra_params_transform = extra_params_transform
        self.target_transform = target_transform
        self.is_3d = is_3d

    def prepare_data(self):
        prepared_train_filepath = self.prepared_data_path / "dataset.hdf5"
        prepared_test_filepath = self.prepared_data_path / "test_dataset.hdf5"
        prepared_eval_filepath = self.prepared_data_path / "eval_dataset.hdf5"

        if self.force_prepare or not check_if_prepared_data_exists(prepared_train_filepath):
            prepare_raw_data(
                self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_train_filepath,
                self.train_cases, self.window_size, self.window_step_size, self.channels, self.extra_params, is_3d=self.is_3d)
        if self.force_prepare or not check_if_prepared_data_exists(prepared_test_filepath):
            prepare_raw_data(
                self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_test_filepath,
                self.test_cases, self.window_size, self.window_step_size, self.channels, self.extra_params, is_3d=self.is_3d)
        if self.eval_cases is not None and (self.force_prepare or not check_if_prepared_data_exists(prepared_eval_filepath)):
            prepare_raw_data(
                self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_eval_filepath,
                self.eval_cases, 1, 1, self.channels, self.extra_params, is_3d=self.is_3d)

    def setup(self, stage: str, split_ratio: float=0.8):
        self.leap_test = LEAP3DDataset(self.prepared_data_path, train=False, extra_params=self.extra_params,
                                       transform=self.tranform, target_transform=self.target_transform, extra_params_tranform=self.extra_params_transform)
        leap_full = LEAP3DDataset(self.prepared_data_path, train=True, extra_params=self.extra_params,
                                  transform=self.tranform, target_transform=self.target_transform, extra_params_tranform=self.extra_params_transform)

        train_points_count = int(np.ceil(len(leap_full) * split_ratio))
        val_points_count = len(leap_full) - train_points_count
        self.leap_train, self.leap_val = random_split(
            leap_full, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
        )
        self.leap_eval = LEAP3DDataset(self.prepared_data_path, train=False, is_eval=True, extra_params=self.extra_params,
                                       transform=self.tranform, target_transform=self.target_transform, extra_params_tranform=self.extra_params_transform)

    def train_dataloader(self):
        return DataLoader(self.leap_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.leap_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.leap_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def eval_dataloader(self):
        return DataLoader(self.leap_eval, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage: str):
        pass
