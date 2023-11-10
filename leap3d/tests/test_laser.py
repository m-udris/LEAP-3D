import logging

import numpy as np

from leap3d.config import DATA_DIR, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SAFETY_OFFSET, MELTING_POINT, TIMESTEP_DURATION, SCALE_DISTANCE_BY, SCALE_TIME_BY
from leap3d.laser import convert_laser_position_to_grid_2d, get_next_laser_position
from leap3d.scanning import ScanResults, ScanParameters


TOLERANCE = 2e-1

def test_get_next_laser_position():
    case_index = 0
    case_filename = DATA_DIR / "case_0000.npz"
    params_filename =  DATA_DIR / "Params.npy"
    rough_coord_filename =  DATA_DIR / "Rough_coord.npz"
    case_params = ScanParameters(params_filename, rough_coord_filename, case_index,
                             x_min=X_MIN, x_max=X_MAX,
                             y_min=Y_MIN, y_max=Y_MAX,
                             z_min=Z_MIN, z_max=Z_MAX,
                             safety_offset=SAFETY_OFFSET,
                             melting_point=MELTING_POINT,
                             timestep_duration=TIMESTEP_DURATION,
                             scale_distance_by=1000,
                             scale_time_by=1000)

    case_results = ScanResults(case_filename, scale_distance_by=1000, scale_time_by=1000)

    for i in range(case_results.total_timesteps - 1):
        laser_data = case_results.get_laser_data_at_timestep(i)
        laser_x_new, laser_y_new = get_next_laser_position(laser_data, case_params)
        laser_x_gt, laser_y_gt = case_results.get_laser_coordinates_at_timestep(i+1)

        logging.debug(f"Laser data: {laser_data[0]}, {laser_data[1]}, {laser_data[4]}, {laser_data[5]}")
        logging.debug(f"Predicted: {laser_x_new}, {laser_y_new}")
        logging.debug(f"Ground Truth: {laser_x_gt}, {laser_y_gt}")
        logging.debug(f"Difference: {abs(laser_x_new - laser_x_gt)} {abs(laser_y_new - laser_y_gt)}")

        assert abs(laser_x_new - laser_x_gt) < TOLERANCE
        assert abs(laser_y_new - laser_y_gt) < TOLERANCE


def test_laser_coversion_to_grid_after_scaling():
    case_filename = DATA_DIR / "case_0000.npz"
    params_filename =  DATA_DIR / "Params.npy"
    rough_coord_filename =  DATA_DIR / "Rough_coord.npz"

    case_params_1 = ScanParameters(params_filename, rough_coord_filename, 0,
                             x_min=X_MIN, x_max=X_MAX,
                             y_min=Y_MIN, y_max=Y_MAX,
                             z_min=Z_MIN, z_max=Z_MAX,
                             safety_offset=SAFETY_OFFSET,
                             melting_point=MELTING_POINT,
                             timestep_duration=TIMESTEP_DURATION,
                             scale_distance_by=1,
                             scale_time_by=1)
    case_params_2 = ScanParameters(params_filename, rough_coord_filename, 0,
                             x_min=X_MIN, x_max=X_MAX,
                             y_min=Y_MIN, y_max=Y_MAX,
                             z_min=Z_MIN, z_max=Z_MAX,
                             safety_offset=SAFETY_OFFSET,
                             melting_point=MELTING_POINT,
                             timestep_duration=TIMESTEP_DURATION,
                             scale_distance_by=1000,
                             scale_time_by=1)

    case_results_1 = ScanResults(case_filename, scale_distance_by=1, scale_time_by=1)
    case_results_2 = ScanResults(case_filename, scale_distance_by=1000, scale_time_by=1)

    for i in range(case_results_1.total_timesteps):
        case_results_1_laser_x, case_results_1_laser_y = case_results_1.get_laser_coordinates_at_timestep(i)
        case_results_2_laser_x, case_results_2_laser_y = case_results_2.get_laser_coordinates_at_timestep(i)

        case_results_1_laser_grid = convert_laser_position_to_grid_2d(case_results_1_laser_x, case_results_1_laser_y, rough_coordinates=case_params_1.rough_coordinates)
        case_results_2_laser_grid = convert_laser_position_to_grid_2d(case_results_2_laser_x, case_results_2_laser_y, rough_coordinates=case_params_2.rough_coordinates)

        assert np.allclose(case_results_1_laser_grid, case_results_2_laser_grid, atol=0.0001)
