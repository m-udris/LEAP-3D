from os import listdir
from os.path import isfile, join
from pathlib import Path
import re
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from LEAP3D.config import DATASETDIR
from LEAP3D.scanning import ScanParameters, ScanResults



# data format: sliding window of 3 (?) timesteps, ground_truth at t+4
# model input: 3D rough temperature + melt pool + laser position at each time step of the window,
#              constant parameters of that simulation


def get_scan_result_filepaths(data_dir: str="path/to/dir", cases: int | List=None):
    if type(cases) == int:
            cases = range(cases)

    filenames = [
        filename for filename in listdir(data_dir)
            if isfile(join(data_dir, filename)) and re.match(r"^case_\d*\.npz$", filename) is not None]
    if cases is not None:
        filenames = filter(lambda filename: int(re.match(r'(\d+)', filename)) in cases, filenames)

        return [Path(data_dir) / filename for filename in filenames]


def get_scan_results_windows(scan_results: ScanResults, window_size: int, step_size: int):
    pass


def save_scan_results_windows(scan_results_windows: [ScanResults], data_dir):
    pass


class LEAP3DDataset(Dataset):
    def __init__(self, data_dir: str="path/to/dir", cases: int | List=None, batch_size: int=32):
        super().__init__()
        pass


class LEAP3DDataModule(pl.LightningDataModule):
    def __init__(self,
                 raw_data_dir: str="path/to/dir", processed_data_dir: str="path/to/dir",
                 batch_size: int=32,
                 cases: int | List=None, window_size: int=3, window_step_size: int=4):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.cases = cases
        self.window_size = window_size
        self.window_step_size = window_step_size


    def prepare_data(self):
        #TODO
        # Split data into the data format described above and save to scratch?
        case_result_filepaths = get_scan_result_filepaths(self.raw_data_dir, self.cases)
        for filepath in case_result_filepaths:
            scan_results = ScanResults(filepath)
            scan_results_windows = get_scan_results_windows(scan_results, self.window_size, self.window_step_size)
            save_scan_results_windows(scan_results_windows, )
        pass

    def setup(self, stage: str, split_ratio: float=0.8):
        self.leap3d_test = LEAP3DDataset(self.data_dir, train=False)
        self.leap3d_predict = LEAP3DDataset(self.data_dir, train=False)
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