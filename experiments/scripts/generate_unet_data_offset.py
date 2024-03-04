from os import mkdir
from multiprocessing import Pool
from pathlib import Path

import h5py

from leap3d.datasets import UNetInterpolationDataModule
from leap3d.datasets.channels import OffsetRoughTemperatureAroundLaser, OffsetTemperatureAroundLaser, ScanningAngle, LaserPower, LaserRadius
from leap3d.config import NUM_WORKERS, DATA_DIR, ROUGH_COORDS_FILEPATH, PARAMS_FILEPATH, DATASET_DIR


def generate_case_dataset(path, case_id, is_test=False, force_prepare=False):
    print(f'Generating dataset for case {case_id}...')
    subfolder_name = ('test_' if is_test else 'train_') + f'case_{case_id}'
    dataset_dir = Path(path) / subfolder_name
    if not dataset_dir.exists():
        mkdir(dataset_dir)


    datamodule = UNetInterpolationDataModule(
        scan_parameters_filepath=PARAMS_FILEPATH,
        rough_coordinates_filepath=ROUGH_COORDS_FILEPATH,
        raw_data_directory=DATA_DIR,
        prepared_data_path=dataset_dir,
        is_3d=False,
        train_cases=[case_id] if not is_test else None,
        test_cases=[case_id] if is_test else None,
        input_shape=[24*4, 24*4],
        target_shape=[24*4, 24*4],
        input_channels=[OffsetRoughTemperatureAroundLaser],
        extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
        target_channels=[OffsetTemperatureAroundLaser],
        transforms={},
        inverse_transforms={},
        include_distances_to_melting_pool=False,
        force_prepare=force_prepare,
        num_workers=NUM_WORKERS
    )
    datamodule.prepare_data('fit' if not is_test else 'test')


def aggregate_datasets(path, cases, is_test=False, force_prepare=False):
    if not Path(path).exists():
        mkdir(path)
    with Pool(min(NUM_WORKERS, len(cases))) as p:
        p.starmap(generate_case_dataset, [(path, case_id, is_test, force_prepare) for case_id in cases])

    filename = ('test_' if is_test else '') + 'dataset.hdf5'
    main_path = Path(path) / filename
    with h5py.File(main_path, 'w') as f:
        for i in cases:
            subfolder_name = ('test_' if is_test else 'train_') + f'case_{i}'
            dataset_dir = Path(path) / subfolder_name
            with h5py.File(dataset_dir / filename, 'r') as subf:
                for k in subf.keys():
                    if k not in f.keys():
                        data = subf[k]
                        f.create_dataset(k, data=data, maxshape=(None, *data.shape[1:]), chunks=True)
                    else:
                        f[k].resize(f[k].shape[0] + subf[k].shape[0], axis=0)
                        f[k][-subf[k].shape[0]:] = subf[k][:]


if __name__ == '__main__':
    train_cases = list(range(18))
    aggregate_datasets(DATASET_DIR / 'unet_interpolation_offset_no_distances', train_cases, False, False)
    test_cases = [18, 19]
    aggregate_datasets(DATASET_DIR / 'unet_interpolation_offset__no_distances', test_cases, False, False)
