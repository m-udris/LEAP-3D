from pathlib import Path

import numpy as np
from scipy import interpolate

from leap3d.config import SCALE_DISTANCE_BY, SCALE_TIME_BY
class ScanResults():
    def __init__(self, results_filename: str | Path,
                 scale_distance_by: float=None,
                 scale_time_by: float=None):
        self.scale_distance_by = scale_distance_by or SCALE_DISTANCE_BY
        self.scale_time_by = scale_time_by or SCALE_TIME_BY


        self.results = np.load(results_filename, allow_pickle=True)
        self.rough_temperature_field = self.results['Rough_T_field']
        self.total_timesteps = self.rough_temperature_field.shape[0]

        self.melt_pool = self.results['Melt_pool']
        self.scale_melt_pool()

        self.laser_data = self.results['Laser_information']
        self.laser_data[:,0] *= self.scale_distance_by
        self.laser_data[:,1] *= self.scale_distance_by
        self.laser_data[:,3] *= self.scale_distance_by
        self.laser_data[:,4] *= self.scale_distance_by / self.scale_time_by
        self.laser_data[:,5] *= self.scale_distance_by / self.scale_time_by


    def scale_melt_pool(self):
        for timestep in range(self.total_timesteps):
            for i in range(len(self.melt_pool[timestep])):
                if len(self.melt_pool[timestep][i]) == 0:
                    continue
                self.melt_pool[timestep][i][0] *= self.scale_distance_by
                self.melt_pool[timestep][i][1] *= self.scale_distance_by
                self.melt_pool[timestep][i][2] *= self.scale_distance_by
                self.melt_pool[timestep][i][3] *= self.scale_distance_by

                self.melt_pool[timestep][i][4] /= self.scale_distance_by
                self.melt_pool[timestep][i][5] /= self.scale_distance_by
                self.melt_pool[timestep][i][6] /= self.scale_distance_by


    def get_rough_temperatures_at_timestep(self, timestep: int):
        return self.rough_temperature_field[timestep]

    def get_top_layer_temperatures(self, timestep: int):
        top_layer_results_at_timestep = self.rough_temperature_field[timestep, :, :, -1]

        return top_layer_results_at_timestep

    def get_laser_coordinates_at_timestep(self, timestep: int):
        """Get laser x and y coordinates at timestep"""
        laser_data = self.laser_data[timestep]
        return laser_data[0], laser_data[1]

    def get_laser_data_at_timestep(self, timestep: int):
        return self.laser_data[timestep]

    def get_laser_power_at_timestep(self, timestep: int):
        return self.laser_data[timestep][2]

    def get_laser_radius_at_timestep(self, timestep: int):
        return self.laser_data[timestep][3]

    def get_melt_pool_coordinates_and_temperature(self, timestep: int):
        coordinates = [point[:3] for point in self.melt_pool[timestep] if len(point) != 0]
        temperatures = [point[3] for point in self.melt_pool[timestep] if len(point) != 0]

        return coordinates, temperatures

    def get_coordinate_points_and_temperature_at_timestep(self, scan_parameters, timestep: int):
        temperature = self.get_rough_temperatures_at_timestep(timestep).flatten()
        melt_pool_coordinates, melt_pool_temperature = self.get_melt_pool_coordinates_and_temperature(timestep)

        coordinates = scan_parameters.get_rough_coordinates_for_cross_section() + melt_pool_coordinates
        temperatures = list(temperature) + melt_pool_temperature

        return coordinates, temperatures

    def get_interpolated_data_along_laser_at_timestep(
            self, scan_parameters, timestep,
            steps=128, method='nearest', return_grid=False
            ):

        coordinates, temperature = self.get_coordinate_points_and_temperature_at_timestep(scan_parameters, timestep)
        laser_x, laser_y = self.get_laser_coordinates_at_timestep(timestep)
        laser_angle_tan = np.tan(scan_parameters.scanning_angle)
        get_y = lambda x: laser_angle_tan * (x - laser_x) + laser_y

        xi = np.linspace(scan_parameters.x_min, scan_parameters.x_max, steps)
        zi = np.linspace(scan_parameters.z_min, scan_parameters.z_max, 64)
        yi = get_y(xi)
        yi_within_bounds_indices = np.where((yi >= scan_parameters.y_min) & (yi <= scan_parameters.y_max))  # Keep elements within bounds
        yi = yi[yi_within_bounds_indices]
        xi = xi[yi_within_bounds_indices]

        cross_section_points = np.array([(x, y, z) for (x, y) in zip(xi, yi) for z in zi])

        T_interpolated = interpolate.griddata(
            coordinates,
            temperature,
            cross_section_points,
            method=method)

        if return_grid:
            return T_interpolated, xi, yi, zi
        return T_interpolated
