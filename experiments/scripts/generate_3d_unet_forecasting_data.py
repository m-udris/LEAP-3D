from os import mkdir
from multiprocessing import Pool
from pathlib import Path

import h5py

from leap3d.datasets import UNetForecastingDataModule
from leap3d.datasets.channels import LaserPosition, RoughTemperature, RoughTemperatureDifference, ScanningAngle, LaserPower, LaserRadius
from leap3d.config import NUM_MP_DATA_PREP_WORKERS, DATA_DIR, ROUGH_COORDS_FILEPATH, PARAMS_FILEPATH, DATASET_DIR


def generate_case_dataset(path, case_id, force_prepare=False, is_eval=False):
    print(f'Generating dataset for case {case_id}...')
    subfolder_prefix = 'eval' if is_eval else 'train'
    subfolder_name = f'{subfolder_prefix}_case_{case_id}'
    dataset_dir = Path(path) / subfolder_name
    if not dataset_dir.exists():
        mkdir(dataset_dir)

    datamodule = UNetForecastingDataModule(
        scan_parameters_filepath=PARAMS_FILEPATH,
        rough_coordinates_filepath=ROUGH_COORDS_FILEPATH,
        raw_data_directory=DATA_DIR,
        prepared_data_path=dataset_dir,
        is_3d=True,
        train_cases=[case_id],
        test_cases=None,
        input_shape=[64, 64, 16],
        target_shape=[64, 64, 16],
        input_channels=[LaserPosition(is_3d=True), RoughTemperature(is_3d=True)],
        extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
        target_channels=[RoughTemperatureDifference(is_3d=True)],
        transforms={},
        inverse_transforms={},
        force_prepare=force_prepare,
        window_size=5 if not is_eval else 1,
        window_step_size=5 if not is_eval else 1,
    )
    datamodule.prepare_data('fit')


def generate_cases(path, cases, force_prepare, eval_cases=[]):
    if not Path(path).exists():
        mkdir(path)
    with Pool(min(NUM_MP_DATA_PREP_WORKERS, len(cases))) as p:
        p.starmap(generate_case_dataset, [(path, case_id, force_prepare, case_id in eval_cases) for case_id in cases])

def aggregate_datasets(path, cases, is_test=False):
    filename = ('test_' if is_test else '') + 'dataset.hdf5'
    main_path = Path(path) / filename

    dataset_shapes = {}

    for i in cases:
        subfolder_name = f'train_case_{i}'
        dataset_dir = Path(path) / subfolder_name
        with h5py.File(dataset_dir / 'dataset.hdf5', 'r') as subf:
            for k in subf.keys():
                if k not in dataset_shapes:
                    dataset_shapes[k] = subf[k].shape
                else:
                    dataset_shapes[k] = (dataset_shapes[k][0] + subf[k].shape[0], *subf[k].shape[1:])

    points_written = {}

    with h5py.File(main_path, 'w') as f:
        for i in cases:
            subfolder_name = f'train_case_{i}'
            dataset_dir = Path(path) / subfolder_name
            with h5py.File(dataset_dir / 'dataset.hdf5', 'r') as subf:
                for k in subf.keys():
                    if k not in f.keys():
                        f.create_dataset(k, shape=dataset_shapes[k])

                    start = points_written.get(k, 0)
                    end = start + subf[k].len()
                    print(f[k].shape, start, subf[k].shape, end)

                    f[k][start:end] = subf[k][:]
                    points_written[k] = end

if __name__ == '__main__':
    num_train_cases = 40
    num_test_cases = 10
    eval_cases = 5
    # train_cases = list(range(num_train_cases)) + list(range(100, 100 + num_train_cases))
    # test_cases = list(range(num_train_cases, num_train_cases + num_test_cases)) + list(range(100 + num_train_cases, 100 + num_train_cases + num_test_cases))
    # eval_cases = list(range(num_train_cases + num_test_cases, num_train_cases + num_test_cases + eval_cases)) + list(range(100 + num_train_cases + num_test_cases, 100 + num_train_cases + num_test_cases + eval_cases))
    train_cases = list(range(0, 200, 2))
    test_cases = list(range(1, 200, 10))
    eval_cases = [5, 35, 65, 95, 105, 135, 165, 195]
    generate_cases(DATASET_DIR / '3d_unet_forecasting', train_cases + test_cases + eval_cases, force_prepare=False, eval_cases=eval_cases)
    aggregate_datasets(DATASET_DIR / '3d_unet_forecasting', train_cases, is_test=False)
    aggregate_datasets(DATASET_DIR / '3d_unet_forecasting', test_cases, is_test=True)
