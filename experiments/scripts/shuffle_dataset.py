from pathlib import Path
import random
import time

import h5py
import numpy as np

from leap3d.config import DATASET_DIR

RANDOM_SEED = 42

def shuffle_time(filepath):
    with h5py.File(filepath, 'r+') as h5f:
        for key in h5f.keys():
            t1 = time.time()
            random.seed(RANDOM_SEED)
            random.shuffle(h5f[key])
            t2 = time.time()
            print(f'Time to shuffle: {t2 - t1:.3f} seconds')

def new_file_shuffle_time(input_path, output_path):
    in_filepath = input_path / 'dataset.hdf5'
    out_filepath = output_path / 'dataset.hdf5'
    with h5py.File(in_filepath, 'r') as h5f_input, h5py.File(out_filepath, 'w') as h5f_output:
        dataset_length = h5f_input['input'].shape[0]
        indices = np.arange(dataset_length)
        t1 = time.time()
        np.random.shuffle(indices)
        t2 = time.time()
        print(f'Time to shuffle: {t2 - t1:.3f} seconds')

        for key in h5f_input.keys():
            t1 = time.time()
            h5f_output.create_dataset(key, data=h5f_input[key], shape=h5f_input[key].shape)
            for index, shuffled_index in enumerate(indices):
                h5f_output[key][index] = h5f_input[key][shuffled_index]
            t2 = time.time()
            print(f'Time to write: {t2 - t1:.3f} seconds')


if __name__ == '__main__':
    input_filepath = Path(DATASET_DIR / 'mlp_interpolation_chunks_gradients_3d_small/')
    output_filepath = Path(DATASET_DIR / 'mlp_interpolation_chunks_gradients_3d_small_shuffled/')
    if not output_filepath.exists():
        output_filepath.mkdir()

    new_file_shuffle_time(input_filepath, output_filepath)
