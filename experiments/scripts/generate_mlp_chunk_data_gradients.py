from os import mkdir
from multiprocessing import Pool
from pathlib import Path

import h5py

from leap3d.datasets import MLPInterpolationChunkDataModule
from leap3d.datasets.channels import LowResRoughTemperatureAroundLaser, MeltPoolPointChunk, ScanningAngle, LaserPower, LaserRadius
from leap3d.config import NUM_MP_DATA_PREP_WORKERS, DATA_DIR, ROUGH_COORDS_FILEPATH, PARAMS_FILEPATH, DATASET_DIR


def generate_case_dataset(path, case_id, force_prepare=False):
    print(f'Generating dataset for case {case_id}...')
    subfolder_name = f'train_case_{case_id}'
    dataset_dir = Path(path) / subfolder_name
    if not dataset_dir.exists():
        mkdir(dataset_dir)

    datamodule = MLPInterpolationChunkDataModule(
        scan_parameters_filepath=PARAMS_FILEPATH,
        rough_coordinates_filepath=ROUGH_COORDS_FILEPATH,
        raw_data_directory=DATA_DIR,
        prepared_data_path=dataset_dir,
        is_3d=False,
        train_cases=[case_id],
        test_cases=None,
        input_shape=[32, 32],
        target_shape=[6],
        input_channels=[LowResRoughTemperatureAroundLaser(is_3d=False, box_size=32, return_coordinates=False)],
        extra_input_channels=[ScanningAngle, LaserPower, LaserRadius],
        target_channels=[MeltPoolPointChunk(is_3d=False, chunk_size=32*32, input_shape=[32,32], include_gradients=True)],
        transforms={},
        inverse_transforms={},
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
    num_train_cases = 50
    num_test_cases = 10
    train_cases = list(range(num_train_cases)) + list(range(100, 100 + num_train_cases))
    test_cases = list(range(num_train_cases, num_train_cases + num_test_cases)) + list(range(100 + num_train_cases, 100 + num_train_cases + num_test_cases))
    generate_cases(DATASET_DIR / 'mlp_interpolation_chunks_gradients', train_cases + test_cases, force_prepare=False)
    aggregate_datasets(DATASET_DIR / 'mlp_interpolation_chunks_gradients', train_cases, is_test=False)
    aggregate_datasets(DATASET_DIR / 'mlp_interpolation_chunks_gradients', test_cases, is_test=True)
