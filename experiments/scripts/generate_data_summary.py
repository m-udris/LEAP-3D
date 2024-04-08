import json
from multiprocessing import Pool
from os import mkdir
from pathlib import Path

import numpy as np

from leap3d.scanning import ScanParameters, ScanResults
from leap3d.config import NUM_MP_DATA_PREP_WORKERS, DATA_DIR, ROUGH_COORDS_FILEPATH, PARAMS_FILEPATH, DATASET_DIR


def bucketize_list(input, bucket_span):
    input = (input // bucket_span) * bucket_span
    unique, counts = np.unique(input, return_counts=True)

    pairs = [(int(item), int(count)) for item, count in zip(unique, counts)]

    return dict(pairs)

def merge_sum_dicts(d1, d2):
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}

def get_case_info(case_index, results_path="./data_summary/"):
    case_filename = DATA_DIR / f"case_{case_index:04}.npz"
    scan_params  = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=case_index)
    scan_results = ScanResults(case_filename)

    # Laser parameter info
    min_laser_power = np.inf
    max_laser_power = -np.inf
    min_laser_radius = np.inf
    max_laser_radius = -np.inf

    # Melt pool info
    min_melt_pool_z = np.inf
    max_melt_pool_z = -np.inf
    min_melt_pool_temperature = np.inf
    max_melt_pool_temperature = -np.inf
    min_melt_pool_grad_x = np.inf
    max_melt_pool_grad_x = -np.inf
    min_melt_pool_grad_y = np.inf
    max_melt_pool_grad_y = -np.inf
    min_melt_pool_grad_z = np.inf
    max_melt_pool_grad_z = -np.inf
    min_melt_pool_grad_t = np.inf
    max_melt_pool_grad_t = -np.inf
    grad_t_bucketized = {}

    # Temperature info
    min_temperature = np.inf
    max_temperature = -np.inf
    temperature_bucketized = {}
    min_temperature_difference = np.inf
    max_temperature_difference = -np.inf
    temperature_difference_bucketized = {}

    total_timesteps = scan_results.total_timesteps

    for t in range(total_timesteps):
        if t % 100 == 0:
            print(f"Processing timestep {t} of {total_timesteps}", end="\r")
        laser_information = scan_results.get_laser_data_at_timestep(t)

        laser_power = laser_information[2]
        laser_radius = laser_information[3]

        min_laser_power = min(min_laser_power, laser_power)
        max_laser_power = max(max_laser_power, laser_power)

        min_laser_radius = min(min_laser_radius, laser_radius)
        max_laser_radius = max(max_laser_radius, laser_radius)

        if not scan_results.is_melt_pool_empty(t):
            melt_pool_data = scan_results.get_melt_pool_data_at_timestep(t)

            melt_pool_z_values = melt_pool_data[:, 2]

            melt_pool_temperatures = melt_pool_data[:, 3]

            melt_pool_grad_x_values = melt_pool_data[:, 4]
            melt_pool_grad_y_values = melt_pool_data[:, 5]
            melt_pool_grad_z_values = melt_pool_data[:, 6]
            melt_pool_grad_t_values = melt_pool_data[:, 7]

            min_melt_pool_z = min(min_melt_pool_z, np.min(melt_pool_z_values))
            max_melt_pool_z = max(max_melt_pool_z, np.max(melt_pool_z_values))

            min_melt_pool_temperature = min(min_melt_pool_temperature, np.min(melt_pool_temperatures))
            max_melt_pool_temperature = max(max_melt_pool_temperature, np.max(melt_pool_temperatures))

            min_melt_pool_grad_x = min(min_melt_pool_grad_x, np.min(melt_pool_grad_x_values))
            max_melt_pool_grad_x = max(max_melt_pool_grad_x, np.max(melt_pool_grad_x_values))

            min_melt_pool_grad_y = min(min_melt_pool_grad_y, np.min(melt_pool_grad_y_values))
            max_melt_pool_grad_y = max(max_melt_pool_grad_y, np.max(melt_pool_grad_y_values))

            min_melt_pool_grad_z = min(min_melt_pool_grad_z, np.min(melt_pool_grad_z_values))
            max_melt_pool_grad_z = max(max_melt_pool_grad_z, np.max(melt_pool_grad_z_values))

            min_melt_pool_grad_t = min(min_melt_pool_grad_t, np.min(melt_pool_grad_t_values))
            max_melt_pool_grad_t = max(max_melt_pool_grad_t, np.max(melt_pool_grad_t_values))

            grad_t_bucketized = merge_sum_dicts(bucketize_list(melt_pool_grad_t_values, 1000), grad_t_bucketized)

        temperature = scan_results.get_rough_temperatures_at_timestep(t)
        temperature = temperature.flatten()

        min_temperature = min(min_temperature, np.min(temperature))
        max_temperature = max(max_temperature, np.max(temperature))

        temperature_bucketized = merge_sum_dicts(bucketize_list(temperature, 10), temperature_bucketized)

        if t < total_timesteps - 1:
            temperature_next = scan_results.get_rough_temperatures_at_timestep(t + 1).flatten()
            temperature_difference = temperature_next - temperature

            min_temperature_difference = min(min_temperature_difference, np.min(temperature_difference))
            max_temperature_difference = max(max_temperature_difference, np.max(temperature_difference))

            temperature_difference_bucketized = merge_sum_dicts(bucketize_list(temperature_difference, 10), temperature_difference_bucketized)

    output = {
        "laser_power": (float(min_laser_power), float(max_laser_power)),
        "laser_radius": (float(min_laser_radius), float(max_laser_radius)),
        "melt_pool_z": (float(min_melt_pool_z), float(max_melt_pool_z)),
        "melt_pool_temperature": (float(min_melt_pool_temperature), float(max_melt_pool_temperature)),
        "melt_pool_grad_x": (float(min_melt_pool_grad_x), float(max_melt_pool_grad_x)),
        "melt_pool_grad_y": (float(min_melt_pool_grad_y), float(max_melt_pool_grad_y)),
        "melt_pool_grad_z": (float(min_melt_pool_grad_z), float(max_melt_pool_grad_z)),
        "melt_pool_grad_t": (float(min_melt_pool_grad_t), float(max_melt_pool_grad_t)),
        "temperature": (float(min_temperature), float(max_temperature)),
        "temperature_bucketized": temperature_bucketized,
        "temperature_difference": (float(min_temperature_difference), float(max_temperature_difference)),
        "temperature_difference_bucketized": temperature_difference_bucketized,
        "grad_t_bucketized": grad_t_bucketized,
    }

    results_filepath = Path(results_path) / f"case_{case_index:04}.json"

    with open(results_filepath, "w") as f:
        json.dump(output, fp=f, indent=4, sort_keys=True)

    return


if __name__ == '__main__':
    result_path = Path("data_summary")
    if not result_path.exists():
        mkdir(result_path)

    cases = range(200)
    with Pool(min(NUM_MP_DATA_PREP_WORKERS, len(cases))) as p:
        p.starmap(get_case_info, [(case_id, result_path) for case_id in cases])
