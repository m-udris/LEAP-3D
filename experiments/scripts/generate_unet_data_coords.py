from os import mkdir
from multiprocessing import Pool
from pathlib import Path

import h5py

from leap3d.datasets import UNetInterpolationDataModule
from leap3d.datasets.channels import RoughTemperatureAroundLaser, TemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius
from leap3d.config import NUM_MP_DATA_PREP_WORKERS, DATA_DIR, ROUGH_COORDS_FILEPATH, PARAMS_FILEPATH, DATASET_DIR


def generate_case_dataset(path, case_id, force_prepare=False):
    print(f'Generating dataset for case {case_id}...')
    subfolder_name = f'train_case_{case_id}'
    dataset_dir = Path(path) / subfolder_name
    if not dataset_dir.exists():
        mkdir(dataset_dir)

    datamodule = UNetInterpolationDataModule(
        scan_parameters_filepath=PARAMS_FILEPATH,
        rough_coordinates_filepath=ROUGH_COORDS_FILEPATH,
        raw_data_directory=DATA_DIR,
        prepared_data_path=dataset_dir,
        is_3d=False,
        train_cases=[case_id],
        test_cases=None,
        input_shape=[128, 128],
        target_shape=[128, 128],
        input_channels=[RoughTemperatureAroundLaser(return_coordinates=True)],
        extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
        target_channels=[TemperatureAroundLaser],
        transforms={},
        inverse_transforms={},
        include_distances_to_melting_pool=False,
        force_prepare=force_prepare
    )
    datamodule.prepare_data('fit')

def generate_cases(path, cases, force_prepare):
    if not Path(path).exists():
        mkdir(path)
    with Pool(min(NUM_MP_DATA_PREP_WORKERS, len(cases))) as p:
        p.starmap(generate_case_dataset, [(path, case_id, force_prepare) for case_id in cases])

def aggregate_datasets(path, cases, is_test=False):
    filename = ('test_' if is_test else '') + 'dataset.hdf5'
    main_path = Path(path) / filename
    with h5py.File(main_path, 'w') as f:
        for i in cases:
            subfolder_name = f'train_case_{i}'
            dataset_dir = Path(path) / subfolder_name
            with h5py.File(dataset_dir / 'dataset.hdf5', 'r') as subf:
                for k in subf.keys():
                    if k not in f.keys():
                        data = subf[k]
                        f.create_dataset(k, data=data, maxshape=(None, *data.shape[1:]), chunks=True)
                    else:
                        f[k].resize(f[k].shape[0] + subf[k].shape[0], axis=0)
                        f[k][-subf[k].shape[0]:] = subf[k][:]


if __name__ == '__main__':
    train_cases = list(range(20)) + list(range(100, 120))
    test_cases = list(range(20, 24)) + list(range(120, 124))

    generate_cases(DATASET_DIR / 'unet_interpolation_coordinates', train_cases + test_cases, force_prepare=True)
    aggregate_datasets(DATASET_DIR / 'unet_interpolation_coordinates', train_cases, is_test=False)
    aggregate_datasets(DATASET_DIR / 'unet_interpolation_coordinates', test_cases, is_test=True)
