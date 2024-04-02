from pathlib import Path

import numpy as np
from scipy import interpolate
import scipy

from leap3d.config import SCALE_DISTANCE_BY, SCALE_TIME_BY, ROUGH_COORDS_STEP_SIZE
from leap3d.scanning import scan_parameters


def custom_interpolate(points, values, desired_points):
    desired_points = np.array(desired_points)
    desired_values = interpolate.interpn(points, values, desired_points, method='linear', bounds_error=False, fill_value=np.nan)
    nan_idxs = np.isnan(desired_values)
    desired_values[nan_idxs] = interpolate.interpn(points, values, desired_points[nan_idxs], method='nearest', bounds_error=False, fill_value=None)
    return desired_values


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

    def get_laser_velocity_at_timestep(self, timestep: int):
        return self.laser_data[timestep][4], self.laser_data[timestep][5]

    def get_melt_pool_data_at_timestep(self, timestep: int, is_3d: bool=True):
        if self.is_melt_pool_empty(timestep):
            return []
        melt_pool_at_timestep = np.array(self.melt_pool[timestep])
        if is_3d:
            return melt_pool_at_timestep

        melt_pool_at_timestep = melt_pool_at_timestep[:,[0,1,2,3,4,5,8]]
        max_z = np.max(melt_pool_at_timestep[:, 2])
        melt_pool_at_timestep = melt_pool_at_timestep[melt_pool_at_timestep[:, 2] == max_z]
        return melt_pool_at_timestep[:,[0,1,3,4,5,6]]

    def get_melt_pool_coordinates_and_temperature(self, timestep: int, is_3d: bool=True):
        if self.is_melt_pool_empty(timestep):
            return [], []
        melt_pool_at_timestep = np.array(self.melt_pool[timestep])
        if is_3d:
            return melt_pool_at_timestep[:, :3], melt_pool_at_timestep[:, 3]

        max_z = np.max(melt_pool_at_timestep[:, 2])
        top_layer_points = melt_pool_at_timestep[melt_pool_at_timestep[:, 2] == max_z]
        return top_layer_points[:, :2], top_layer_points[:, 3]

    def get_melt_pool_coordinates_and_gradients(self, timestep: int, is_3d: bool=True):
        if self.is_melt_pool_empty(timestep):
            return [], []
        melt_pool_at_timestep = np.array(self.melt_pool[timestep])
        if is_3d:
            return melt_pool_at_timestep[:, :3], melt_pool_at_timestep[:, [4,5,6,8]]

        max_z = np.max(melt_pool_at_timestep[:, 2])
        top_layer_points = melt_pool_at_timestep[melt_pool_at_timestep[:, 2] == max_z]
        return top_layer_points[:, :2], top_layer_points[:, [4,5,6,8]]

    def is_melt_pool_empty(self, timestep: int):
        return len(self.melt_pool[timestep][0]) == 0

    def _get_high_res_grid(self, coordinates, values):
        grid = np.zeros((256, 256, 64, *values.shape[1:]))

        if len(coordinates) == 0:
            return grid

        coord_delta = ROUGH_COORDS_STEP_SIZE * 0.25
        new_x_coords = [int((coord[0]) / coord_delta) + 128  for coord in coordinates]
        new_y_coords = [int((coord[1]) / coord_delta) + 128 for coord in coordinates]
        new_z_coords = [int((coord[2]) / coord_delta) + 63  for coord in coordinates]

        for i in range(len(coordinates)):
            grid[new_x_coords[i]][new_y_coords[i]][new_z_coords[i]] = values[i]

        return grid

    def get_melt_pool_temperature_grid(self, timestep: int):
        coordinates, temperature = self.get_melt_pool_coordinates_and_temperature(timestep)
        grid = self._get_high_res_grid(coordinates, temperature)
        grid.reshape(grid.shape[:-1])

    def get_gradients_grid(self, timestep: int):
        coordinates, gradients = self.get_melt_pool_coordinates_and_gradients(timestep)
        return self._get_high_res_grid(coordinates, gradients)

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
                                                is_3d: bool=False,
                                                offset_ratio: float=None):
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

        if offset_ratio is None:
            points_to_interpolate = scan_parameters.get_coordinates_around_position(laser_x, laser_y, size,
                                                                                scale=step_scale, is_3d=is_3d)
        else:
            points_to_interpolate = scan_parameters.get_offset_coordinates_around_position(self.get_laser_data_at_timestep(timestep), size, step_scale, offset_ratio, is_3d)

        points_to_interpolate = np.array(points_to_interpolate)

        if include_high_resolution_points:
            high_res_points, high_res_temps = self.get_melt_pool_coordinates_and_temperature(timestep)
            if len(high_res_points) == 0:
                interpolated_values = custom_interpolate(grid_points, rough_temperature, points_to_interpolate)
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
                upscaled_rough_temperatures = custom_interpolate(grid_points, rough_temperature, upscaled_rough_coordinates)
                upscaled_rough_temperatures = upscaled_rough_temperatures.reshape(upscaled_x_range.shape[0], upscaled_y_range.shape[0], upscaled_z_range.shape[0])

                high_res_temps = high_res_temps.reshape((-1,1))
                high_res_data = np.concatenate([high_res_points, high_res_temps], axis=-1)

                new_x_min = upscaled_x_range[0]
                new_x_max = upscaled_x_range[-1]

                new_y_min = upscaled_y_range[0]
                new_y_max = upscaled_y_range[-1]

                new_z_min = upscaled_z_range[0]

                high_res_data = np.array(list(filter(lambda x: (new_x_min <= x[0] <= new_x_max) and (new_y_min <= x[1] <= new_y_max) and (new_z_min <= x[2]), high_res_data)))

                if high_res_data.shape[0] != 0:
                    high_res_data[:, :3] = np.round((high_res_data[:, :3] - np.array([new_x_min, new_y_min, new_z_min])) / step_size)
                    high_res_data[:, 2] = np.array([-1 if z >= upscaled_rough_temperatures.shape[2] else z for z in high_res_data[:, 2]]).astype(int)
                    upscaled_rough_temperatures[high_res_data[:, 0].astype(int), high_res_data[:, 1].astype(int), high_res_data[:, 2].astype(int)] = np.max((high_res_data[:, 3], upscaled_rough_temperatures[high_res_data[:, 0].astype(int), high_res_data[:, 1].astype(int), high_res_data[:, 2].astype(int)]), axis=0)

                upscaled_grid_points = [upscaled_x_range, upscaled_y_range, upscaled_z_range]
                interpolated_values = custom_interpolate(upscaled_grid_points, upscaled_rough_temperatures, points_to_interpolate)

        else:
            interpolated_values = custom_interpolate(grid_points, rough_temperature, points_to_interpolate)

        points_to_interpolate = np.array(points_to_interpolate)
        edge_length = int(size // step_scale)
        shape = (edge_length, edge_length, 64) if is_3d else (edge_length, edge_length)
        x_coords = points_to_interpolate[:, 0].reshape(shape)
        y_coords = points_to_interpolate[:, 1].reshape(shape)
        z_coords = points_to_interpolate[:, 2].reshape(shape)
        interpolated_values = interpolated_values.reshape(shape)

        return (x_coords, y_coords, z_coords), interpolated_values

    def sample_interpolated_points_within_bounds(self, scan_parameters, timestep, bounds, size: int, is_3d: bool=False):
        points_to_interpolate = scan_parameters.sample_coordinates_within_bounds(bounds, size, is_3d)
        points_to_interpolate = np.array(points_to_interpolate)

        # coordinates, temperatures = self.get_coordinate_points_and_temperature_at_timestep(scan_parameters, timestep)
        rough_coordinates = scan_parameters.rough_coordinates
        x_rough = rough_coordinates['x_rough']
        y_rough = rough_coordinates['y_rough']
        z_rough = rough_coordinates['z_rough']
        grid_points = ([x[0][0] for x in x_rough], [y[0] for y in y_rough[0]], [z for z in z_rough[0][0]])

        rough_temperature = self.get_rough_temperatures_at_timestep(timestep)


        high_res_points, high_res_temps = self.get_melt_pool_coordinates_and_temperature(timestep)
        if len(high_res_points) == 0:
            interpolated_values = custom_interpolate(grid_points, rough_temperature, points_to_interpolate)
        else:
            step_size = scan_parameters.rough_coordinates_step_size * 0.25

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
            upscaled_rough_temperatures = custom_interpolate(grid_points, rough_temperature, upscaled_rough_coordinates)
            upscaled_rough_temperatures = upscaled_rough_temperatures.reshape(upscaled_x_range.shape[0], upscaled_y_range.shape[0], upscaled_z_range.shape[0])

            high_res_temps = high_res_temps.reshape((-1,1))
            high_res_data = np.concatenate([high_res_points, high_res_temps], axis=-1)

            new_x_min = upscaled_x_range[0]
            new_x_max = upscaled_x_range[-1]

            new_y_min = upscaled_y_range[0]
            new_y_max = upscaled_y_range[-1]

            new_z_min = upscaled_z_range[0]

            high_res_data = np.array(list(filter(lambda x: (new_x_min <= x[0] <= new_x_max) and (new_y_min <= x[1] <= new_y_max) and (new_z_min <= x[2]), high_res_data)))

            if high_res_data.shape[0] != 0:
                high_res_data[:, :3] = np.round((high_res_data[:, :3] - np.array([new_x_min, new_y_min, new_z_min])) / step_size)
                high_res_data[:, 2] = np.array([-1 if z >= upscaled_rough_temperatures.shape[2] else z for z in high_res_data[:, 2]]).astype(int)
                upscaled_rough_temperatures[high_res_data[:, 0].astype(int), high_res_data[:, 1].astype(int), high_res_data[:, 2].astype(int)] = np.max((high_res_data[:, 3], upscaled_rough_temperatures[high_res_data[:, 0].astype(int), high_res_data[:, 1].astype(int), high_res_data[:, 2].astype(int)]), axis=0)

            upscaled_grid_points = [upscaled_x_range, upscaled_y_range, upscaled_z_range]
            interpolated_values = custom_interpolate(upscaled_grid_points, upscaled_rough_temperatures, points_to_interpolate)

        return points_to_interpolate, interpolated_values
