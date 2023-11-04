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


def check_if_prepared_data_exists(filepath: str| Path):
    # Might be a good idea to add additional checks, e.g. whether
    # dataset metadata matches
    return isfile(filepath)


def prepare_raw_data(scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                     raw_data_dir: str | Path, prepared_data_path: str| Path, cases: int | List=None):
    # TODO: Refactor
    params_file = np.load(scan_parameters_filepath)
    rough_coordinates = np.load(rough_coordinates_filepath)

    offset = 0
    with h5py.File(prepared_data_path, 'w') as f:
        train_dset = f.create_dataset("train", (1000, 64, 64, 3), maxshape=(None, 64, 64, 3), chunks=True)
        target_dset = f.create_dataset("target", (1000, 64, 64), maxshape=(None, 64, 64), chunks=True)
        scan_result_filepaths = get_scan_result_case_ids_and_filepaths(raw_data_dir, cases)
        for case_id, scan_result_filepath in scan_result_filepaths:
            scan_parameters = ScanParameters(params_file, rough_coordinates, case_id,
                                            X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SAFETY_OFFSET,
                                            MELTING_POINT, TIMESTEP_DURATION)

            training_points, targets = prepare_scan_results(scan_result_filepath, scan_parameters)
            number_of_points = len(training_points)
            if train_dset.len() < offset + number_of_points:
                train_dset.resize(offset + number_of_points, axis=0)
                target_dset.resize(offset + number_of_points, axis=0)
            train_dset[offset:] = training_points
            train_dset.flush()
            target_dset[offset:] = targets
            target_dset.flush()

            offset += number_of_points


def prepare_scan_results(scan_result_filepath, scan_parameters, dims=2):
    scan_results = ScanResults(scan_result_filepath)

    training_points = []
    targets = []

    for timestep in range(0, scan_results.total_timesteps - 1):

        temperatures = scan_results.rough_temperature_field[timestep, :, :, -1]
        next_step_temperatures =  scan_results.rough_temperature_field[timestep+1, :, :, -1]

        laser_data = scan_results.get_laser_data_at_timestep(timestep)
        next_laser_data = scan_results.get_laser_data_at_timestep(timestep + 1)

        laser_position_grid = convert_laser_position_to_grid(laser_data[0], laser_data[1], scan_parameters.rough_coordinates, dims)
        next_laser_position_grid = convert_laser_position_to_grid(next_laser_data[0], next_laser_data[1], scan_parameters.rough_coordinates, dims)

        training_points.append(
            np.array([laser_position_grid, next_laser_position_grid, temperatures]))
        next_step_diffferences = next_step_temperatures - temperatures
        targets.append(next_step_diffferences)

    training_points = np.array(training_points)
    shape = training_points.shape
    training_points = training_points.reshape((shape[0], shape[2], shape[3], shape[1]))

    return training_points, np.array(targets)


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
                 train: bool=True,
                 transform=None, target_transform=None):
        super().__init__()
        self.train = train
        self.data_filepath = Path(data_dir) / ("dataset.hdf5" if train else "test_dataset.hdf5")
        self.data_file = h5py.File(self.data_filepath, 'r')
        self.x_train = self.data_file['train']
        self.targets = self.data_file['target']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.x_train.len()

    def __getitem__(self, idx):
        data = self.x_train[idx]
        target = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target

    def __del__(self):
        self.data_file.close()


class LEAP3DDataModule(pl.LightningDataModule):
    def __init__(self,
                 scan_parameters_filepath: str | Path, rough_coordinates_filepath: str | Path,
                 raw_data_dir: str="path/to/dir", prepared_data_path: str="path/to/dir",
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 window_size: int=1, window_step_size: int=1,
                 transform: callable=None, target_transform: callable=None,
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
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.force_prepare = force_prepare
        self.num_workers = num_workers
        self.tranform = transform
        self.target_transform = target_transform

    def prepare_data(self):
        prepared_train_filepath = self.prepared_data_path / "dataset.hdf5"
        prepared_test_filepath = self.prepared_data_path / "test_dataset.hdf5"
        if self.force_prepare or not check_if_prepared_data_exists(prepared_train_filepath):
            prepare_raw_data(self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_train_filepath, self.train_cases)
        if self.force_prepare or not check_if_prepared_data_exists(prepared_test_filepath):
            prepare_raw_data(self.scan_parameters_filepath, self.rough_coordinates_filepath, self.raw_data_dir, prepared_test_filepath, self.test_cases)

    def setup(self, stage: str, split_ratio: float=0.8):
        self.leap_test = LEAP3DDataset(self.prepared_data_path, train=False,
                                       transform=self.tranform, target_transform=self.target_transform)
        leap_full = LEAP3DDataset(self.prepared_data_path, train=True,
                                  transform=self.tranform, target_transform=self.target_transform)

        train_points_count = int(np.ceil(len(leap_full) * split_ratio))
        val_points_count = len(leap_full) - train_points_count
        self.leap_train, self.leap_val = random_split(
            leap_full, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.leap_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.leap_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.leap_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.leap_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        pass
