from pathlib import Path

import numpy as np
from scipy import interpolate
import scipy

from leap3d.config import SCALE_DISTANCE_BY, SCALE_TIME_BY
from leap3d.scanning import scan_parameters
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

    def get_melt_pool_temperature_grid(self, timestep: int):
        melt_pool_grid = np.zeros((256, 256, 64))
        coordinates, temperature = self.get_melt_pool_coordinates_and_temperature(timestep)
        if len(coordinates) == 0:
            return melt_pool_grid

        x_coords = set([coord[0] for coord in coordinates])
        sorted_x_coords = sorted(list(x_coords))
        coord_delta = list(set([sorted_x_coords[i+1] - sorted_x_coords[i] for i in range(len(sorted_x_coords) - 1)]))[0]

        new_x_coords = [int((coord[0]) / coord_delta) + 128  for coord in coordinates]
        new_y_coords = [int((coord[1]) / coord_delta) + 128 for coord in coordinates]
        new_z_coords = [int((coord[2]) / coord_delta) + 63  for coord in coordinates]

        for i in range(len(coordinates)):
            melt_pool_grid[new_x_coords[i]][new_y_coords[i]][new_z_coords[i]] = temperature[i]

        return melt_pool_grid

    def get_coordinate_points_and_temperature_at_timestep(self, scan_parameters, timestep: int, resolution: str=None):
        if resolution not in [None, 'low', 'high']:
            raise ValueError("Resolution must be one of [None, 'low', 'high']")

        if resolution is None or resolution == 'low':
            low_res_coordinates = scan_parameters.get_rough_coordinates_points_list()
            low_res_temperature = list(self.get_rough_temperatures_at_timestep(timestep).flatten())
        if resolution == 'low':
            return low_res_coordinates, low_res_temperature

        if resolution is None or resolution == 'high':
            high_res_coordinates, high_res_temperature = self.get_melt_pool_coordinates_and_temperature(timestep)

        if resolution == 'high':
            return high_res_coordinates, high_res_temperature

        coordinates = []
        temperatures = []

        for point, value in zip(low_res_coordinates, low_res_temperature):
            if point not in high_res_coordinates:
                coordinates.append(point)
                temperatures.append(value)

        coordinates.extend(high_res_coordinates)
        temperatures.extend(high_res_temperature)

        return coordinates, temperatures

    def get_interpolated_data_along_laser_at_timestep(
            self, scan_parameters, timestep,
            steps=128, method='nearest', return_grid=False,
            use_laser_position=None
            ):

        coordinates, temperature = self.get_coordinate_points_and_temperature_at_timestep(scan_parameters, timestep)
        if use_laser_position is None:
            laser_x, laser_y = self.get_laser_coordinates_at_timestep(timestep)
        else:
            laser_x, laser_y = use_laser_position
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

    def get_interpolated_data(self, scan_parameters, timestep, shape=(256, 256, 64), method='nearest'):
        # TODO: fix so that it's aligned to the correct grid
        coordinates, temperature = self.get_coordinate_points_and_temperature_at_timestep(scan_parameters, timestep)

        xi = np.linspace(scan_parameters.x_min, scan_parameters.x_max, shape[0])
        yi = np.linspace(scan_parameters.y_min, scan_parameters.y_max, shape[1])
        zi = np.linspace(scan_parameters.z_min, scan_parameters.z_max, shape[2])

        new_coordinate_points = np.array([(x, y, z) for x in xi for y in yi for z in zi])
        temperature = scipy.interpolate.griddata(coordinates, temperature, new_coordinate_points, method=method)
        temperature = temperature.reshape(shape)

        return new_coordinate_points, temperature

    def get_interpolated_grid_around_laser(self, timestep: int,
                                                size: int, step_scale: float,
                                                scan_parameters: scan_parameters,
                                                include_high_resolution_points: bool=False,
                                                is_3d: bool=False):
        """Get interpolated grid around laser at timestep.

        Args:
            timestep (int): Timestep to get grid around laser
            size (int): Square edge length in amount of points of rough coordinates around laser
            step_scale (float): Scale of new step size compared to rough coordinates. E.g. 0.25 means 4 times as many points
            scan_parameters (scan_parameters): Scan parameters
            include_high_res (bool, optional): Whether to include high res points. Defaults to False.
            is_3d (bool, optional): Whether to get 3D grid. Defaults to False."""
        laser_x, laser_y = self.get_laser_coordinates_at_timestep(timestep)
        rough_coordinates = scan_parameters.rough_coordinates
        x_rough = rough_coordinates['x_rough']
        y_rough = rough_coordinates['y_rough']
        z_rough = rough_coordinates['z_rough']
        grid_points = ([x[0][0] for x in x_rough], [y[0] for y in y_rough[0]], [z for z in z_rough[0][0]])

        rough_temperature = self.get_rough_temperatures_at_timestep(timestep)

        points_to_interpolate = scan_parameters.get_coordinates_around_position(laser_x, laser_y, size,
                                                                                scale=step_scale, is_3d=is_3d)

        points_to_interpolate = np.array(points_to_interpolate)

        if include_high_resolution_points:
            high_res_points, high_res_temps = self.get_melt_pool_coordinates_and_temperature(timestep)
            if len(high_res_points) == 0:
                interpolated_values = interpolate.interpn(grid_points, rough_temperature, points_to_interpolate, method='linear')
            else:
                step_size = scan_parameters.rough_coordinates_step_size * step_scale

                # get ranges of x and y and align them to the new grid
                half_step = step_size / 2
                x_min = np.min(x_rough) - half_step
                x_max = np.max(x_rough) - half_step
                y_min = np.min(y_rough) + half_step
                y_max = np.max(y_rough) - half_step
                # We don't need to align z
                z_min = np.min(z_rough)
                z_max = np.max(z_rough)

                upscaled_x_range = np.arange(x_min, x_max, step_size)
                desired_x_min = np.min(points_to_interpolate[:, 0])
                desired_x_max = np.max(points_to_interpolate[:, 0])
                upscaled_x_range = upscaled_x_range[(upscaled_x_range >= desired_x_min - step_size) & (upscaled_x_range <= desired_x_max + step_size)]

                upscaled_y_range = np.arange(y_min, y_max, step_size)
                desired_y_min = np.min(points_to_interpolate[:, 1])
                desired_y_max = np.max(points_to_interpolate[:, 1])
                upscaled_y_range = upscaled_y_range[(upscaled_y_range >= desired_y_min - step_size) & (upscaled_y_range <= desired_y_max + step_size)]

                z_min = z_min if is_3d else z_max - 3 * step_size
                upscaled_z_range = np.arange(z_min, z_max + step_size, step_size)  # z coords are <= 0

                upscaled_rough_coordinates = [(x, y, z) for x in upscaled_x_range for y in upscaled_y_range for z in upscaled_z_range]
                upscaled_rough_temperatures = interpolate.interpn(grid_points, rough_temperature, upscaled_rough_coordinates, method='linear', bounds_error=False, fill_value=None)
                upscaled_rough_temperatures = upscaled_rough_temperatures.reshape(upscaled_x_range.shape[0], upscaled_y_range.shape[0], upscaled_z_range.shape[0])

                for high_res_point, high_res_temp in zip(high_res_points, high_res_temps):
                    x, y, z = high_res_point
                    if upscaled_z_range[0] - z > step_size:
                        continue
                    x_diff = np.abs(upscaled_x_range - x)
                    if np.min(x_diff) > step_size:
                        continue
                    x_index = np.argmin(x_diff)
                    y_diff = np.abs(upscaled_y_range - y)
                    if np.min(y_diff) > step_size:
                        continue
                    y_index = np.argmin(y_diff)
                    z_index = np.argmin(np.abs(upscaled_z_range - z))
                    upscaled_rough_temperatures[x_index, y_index, z_index] = high_res_temp

                upscaled_grid_points = [upscaled_x_range, upscaled_y_range, upscaled_z_range]
                interpolated_values = interpolate.interpn(upscaled_grid_points, upscaled_rough_temperatures, points_to_interpolate, method='linear', bounds_error=False, fill_value=None)
        else:
            interpolated_values = interpolate.interpn(grid_points, rough_temperature, points_to_interpolate, method='linear')

        points_to_interpolate = np.array(points_to_interpolate)
        edge_length = int(size // step_scale)
        shape = (edge_length, edge_length, 64) if is_3d else (edge_length, edge_length)
        x_coords = points_to_interpolate[:, 0].reshape(shape)
        y_coords = points_to_interpolate[:, 1].reshape(shape)
        z_coords = points_to_interpolate[:, 2].reshape(shape)
        interpolated_values = interpolated_values.reshape(shape)

        return (x_coords, y_coords, z_coords), interpolated_values

    # def get_points_around_laser(self, timestep: int, size: int, scan_parameters: scan_parameters, include_high_res: bool=False, is_3d: bool=False):
    #     # TODO: do we need this?
    #     raise NotImplementedError
    #     # Step 1. Interpolate low res to high res
    #     # Step 2. if include high res, add high res points to low res grid
    #     # Step 3. Get points around laser. Do this by masking the grid according to distance to laser in x and y axes, then remove padding

    #     laser_x, laser_y = self.get_laser_coordinates_at_timestep(timestep)


