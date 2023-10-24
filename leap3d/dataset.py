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

from leap3d.config import DATA_DIR
from leap3d.scanning import ScanParameters, ScanResults
from leap3d.laser import convert_laser_position_to_grid


def check_if_prepared_data_exists(filepath: str| Path):
    # Might be a good idea to add additional checks, e.g. whether
    # dataset metadata matches
    return isfile(filepath)


def prepare_raw_data(raw_data_dir: str | Path, prepared_data_path: str| Path, cases: int | List=None):
    # TODO: Refactor
    offset = 0
    with h5py.File(prepared_data_path, 'w') as f:
        train_dset = f.create_dataset("train", (1000, 64, 64, 3), maxshape=(None, 64, 64, 3), chunks=True)
        target_dset = f.create_dataset("target", (1000, 64, 64), maxshape=(None, 64, 64), chunks=True)
        scan_result_filepaths = get_scan_result_filepaths(raw_data_dir, cases)
        for scan_result_filepath in scan_result_filepaths:
            training_points, targets = prepare_scan_results(scan_result_filepath)
            number_of_points = len(training_points)
            if train_dset.len() < offset + number_of_points:
                train_dset.resize(offset + number_of_points, axis=0)
                target_dset.resize(offset + number_of_points, axis=0)
            train_dset[offset:] = training_points
            train_dset.flush()
            target_dset[offset:] = targets
            target_dset.flush()

            offset += number_of_points


def prepare_scan_results(scan_result_filepath, dims=2):
    # TODO: refactor rough_coord
    rough_coord_filename =  DATA_DIR / "Rough_coord.npz"
    scan_results = ScanResults(scan_result_filepath, rough_coord_filename)
    rough_coordinates = scan_results.rough_coordinates

    training_points = []
    targets = []

    for timestep in range(0, scan_results.total_timesteps - 1):
        temperatures = scan_results.rough_temperature_field[timestep, :, :, -1]
        next_step_temperatures =  scan_results.rough_temperature_field[timestep+1, :, :, -1]

        laser_data = scan_results.get_laser_data_at_timestep(timestep)
        next_laser_data = scan_results.get_laser_data_at_timestep(timestep + 1)

        laser_position_grid = convert_laser_position_to_grid(laser_data[0], laser_data[1], rough_coordinates, dims)
        next_laser_position_grid = convert_laser_position_to_grid(next_laser_data[0], next_laser_data[1], rough_coordinates, dims)

        training_points.append(
            np.array([laser_position_grid, next_laser_position_grid, temperatures]))
        targets.append(next_step_temperatures)

    training_points = np.array(training_points)
    shape = training_points.shape
    training_points = training_points.reshape((shape[0], shape[2], shape[3], shape[1]))

    return training_points, np.array(targets)

# data format: sliding window of 3 (?) timesteps, ground_truth at t+4
# model input: 3D rough temperature + melt pool + laser position at each time step of the window,
#              constant parameters of that simulation


def get_scan_result_filepaths(data_dir: str="path/to/dir", cases: int | List=None):
    if type(cases) == int:
        cases = range(cases)

    filenames = [
        filename for filename in listdir(data_dir)
            if isfile(join(data_dir, filename)) and re.match(r"^case_\d*\.npz$", filename) is not None]
    print(filenames, re.search(r'(\d+)', filenames[0]).group(0))
    if cases is not None:
        filenames = filter(lambda filename: int(re.search(r'(\d+)', filename).group(0) or -1) in cases, filenames)

    return [Path(data_dir) / filename for filename in filenames]


class LEAP3DDataset(Dataset):
    def __init__(self, data_dir: str="path/to/dir",
                 train: bool=True,
                 transform=None, target_transform=None):
        super().__init__()
        self.train = train
        self.data_filepath = Path(data_dir) / "dataset.hdf5" if train else "test_dataset.hdf5"
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
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return data, target

    def __del__(self):
        self.data_file.close()


class LEAP3DDataModule(pl.LightningDataModule):
    def __init__(self,
                 raw_data_dir: str="path/to/dir", prepared_data_path: str="path/to/dir",
                 batch_size: int=32,
                 train_cases: int | List=None, test_cases: int | List=None,
                 window_size: int=1, window_step_size: int=1,
                 force_prepare=False):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.prepared_data_path = prepared_data_path
        self.batch_size = batch_size
        self.train_cases = train_cases
        self.test_cases = test_cases
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.force_prepare = force_prepare

    def prepare_data(self):
        prepared_train_filepath = self.prepared_data_path / "dataset.hdf5"
        prepared_test_filepath = self.prepared_data_path / "test_dataset.hdf5"
        if self.force_prepare or not check_if_prepared_data_exists(prepared_train_filepath):
            prepare_raw_data(self.raw_data_dir, prepared_train_filepath, self.train_cases)
        if self.force_prepare or not check_if_prepared_data_exists(prepared_test_filepath):
            prepare_raw_data(self.raw_data_dir, prepared_test_filepath, self.test_cases)

    def setup(self, stage: str, split_ratio: float=0.8):
        self.leap3d_test = LEAP3DDataset(self.data_dir, train=False)
        leap3d_full = LEAP3DDataset(self.data_dir, train=True)

        train_points_count = torch.ceil(len(leap3d_full) * split_ratio)
        val_points_count = len(leap3d_full) - train_points_count
        self.leap3d_train, self.leap3d_val = random_split(
            leap3d_full, [train_points_count, val_points_count], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.leap_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.leap_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.leap_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.leap_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        pass
