from pathlib import Path
import time
import numpy as np
from leap3d.plotting import get_projected_cross_section

from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.scanning.scan_results import ScanResults
from leap3d.config import DATA_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH
from leap3d.contour import get_melting_pool_contour_2d, get_melting_pool_contour_3d


def precompute_values(case_id: int, scan_results: ScanResults, scan_parameters: ScanParameters, path: Path | str, t_start: int, steps: int, save_every_n_steps: int = 100):
    def save_data(case_id, timestep):
        np.save(path / f"case_{case_id:04d}_interpolated_3d_data_{timestep:04d}.npy", np.array(interpolated_3d_data))
        # with open(path / f"case_{case_id:04d}_contours_2d_{timestep:04d}.txt", 'w') as f:
        #     f.write(contours_2d_list.__repr__())
        #     f.flush()
        # with open(path / f"case_{case_id:04d}_contours_3d_{timestep:04d}.txt", 'w') as f:
        #     f.write(contours_3d_list.__repr__())
        #     f.flush()
        np.save(path / f"case_{case_id:04d}_laser_coordinates_{timestep:04d}.npy", np.array(laser_coordinates_list))
        np.save(path / f"case_{case_id:04d}_cross_sections_{timestep:04d}.npy", interpolated_cross_sections)

    interpolated_3d_data = []
    laser_coordinates_list = []
    interpolated_cross_sections = []
    for timestep in range(t_start, t_start + steps):
        print(timestep)
        coordinates, temperature = scan_results.get_interpolated_data(scan_parameters, timestep)
        interpolated_3d_data.append(temperature)

        laser_coordinates = scan_results.get_laser_coordinates_at_timestep(timestep)
        x_min = min(coordinates, key=lambda x: x[0])[0]
        y_min = min(coordinates, key=lambda x: x[1])[1]
        x_max = max(coordinates, key=lambda x: x[0])[0]
        y_max = max(coordinates, key=lambda x: x[1])[1]
        x_step_size = (x_max - x_min) / 256
        y_step_size = (y_max - y_min) / 256
        laser_x = (laser_coordinates[0] - x_min) / x_step_size
        laser_y = (laser_coordinates[1] - y_min) / y_step_size

        laser_coordinates_list.append([laser_x, laser_y])

        # a = time.perf_counter()
        # contours_2d = get_melting_pool_contour_2d(temperature, top_k=24)
        # b = time.perf_counter()
        # print(f"Time for 2d contour: {b - a}")
        # contours_2d = [[x.tolist() for x in contour_list] for contour_list in contours_2d]
        # contours_2d_list.append(contours_2d)

        # a = time.perf_counter()
        # vertices_3d, faces_3d = get_melting_pool_contour_3d(temperature, top_k=24)
        # b = time.perf_counter()
        # print(f"Time for 3d contour: {b - a}")

        # vertices_3d = [x.tolist() for x in vertices_3d]
        # faces_3d = [x.tolist() for x in faces_3d]
        # contours_3d_list.append((vertices_3d, faces_3d))

        interpolated_cross_section = get_projected_cross_section(scan_results, scan_parameters, timestep)
        interpolated_cross_sections.append(interpolated_cross_section)

        if len(interpolated_3d_data) == save_every_n_steps:
            save_data(case_id, (timestep - t_start) // save_every_n_steps)
            interpolated_3d_data = []
            interpolated_cross_sections = []
            laser_coordinates_list = []

    if len(interpolated_3d_data) > 0:
        save_data(case_id, (t_start + steps) // save_every_n_steps)


if __name__ == '__main__':
    for i in range(1, 3):
        scan_results = ScanResults(DATA_DIR / f'case_{i:04d}.npz')
        scan_parameters = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=i)

        t_start = 2000
        steps = 300

        precompute_values(i, scan_results, scan_parameters, Path("/scratch/snx3000/mudris/leap3d/contours/"), t_start, steps, save_every_n_steps=100)

