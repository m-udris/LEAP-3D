import logging

from leap3d.config import DATA_DIR, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SAFETY_OFFSET, MELTING_POINT, TIMESTEP_DURATION
from leap3d.laser import get_next_laser_position
from leap3d.scanning import ScanResults, ScanParameters


TOLERANCE = 1e-5

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
                             timestep_duration=TIMESTEP_DURATION)

    case_results = ScanResults(case_filename)

    for i in range(case_results.total_timesteps):
        laser_data = case_results.get_laser_data_at_timestep(i)
        laser_x_new, laser_y_new = get_next_laser_position(laser_data, case_params)
        laser_x_gt, laser_y_gt = case_results.get_laser_coordinates_at_timestep(i+1)

        logging.debug(f"Laser data: {laser_data[0]}, {laser_data[1]}, {laser_data[2]}, {laser_data[3]}")
        logging.debug(f"Predicted: {laser_x_new}, {laser_y_new}")
        logging.debug(f"Ground Truth: {laser_x_gt}, {laser_y_gt}")
        logging.debug(f"Difference: {abs(laser_x_new - laser_x_gt)} {abs(laser_y_new - laser_y_gt)}")
        # print(laser_data)
        # print(laser_x_new, laser_y_new, type(laser_x_new))
        # print(laser_x_gt, laser_y_gt, type(laser_x_gt))
        # print(laser_x_new - laser_x_gt, laser_y_new - laser_y_gt)
        assert abs(laser_x_new - laser_x_gt) < TOLERANCE
        assert abs(laser_y_new - laser_y_gt) < TOLERANCE
