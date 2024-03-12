import numpy as np

from leap3d.laser import convert_laser_position_to_grid


class Channel:
    def __init__(self, name: str, channels: int, is_grid: bool):
        self.name = name
        self.channels = channels

    def __str__(self):
        return f'{self.name}'

    def get_channels(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must implement this method')


class LaserPosition(Channel):
    def __init__(self, is_3d=False):
        super().__init__('laser_position', 2, True)
        self.is_3d = is_3d

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        laser_data = scan_results.get_laser_data_at_timestep(timestep)
        next_laser_data = scan_results.get_laser_data_at_timestep(timestep + 1)

        laser_position_grid = convert_laser_position_to_grid(laser_data[0], laser_data[1], scan_parameters.rough_coordinates, is_3d=self.is_3d)
        next_laser_position_grid = convert_laser_position_to_grid(next_laser_data[0], next_laser_data[1], scan_parameters.rough_coordinates, is_3d=self.is_3d)

        return [laser_position_grid, next_laser_position_grid]


class RoughTemperature(Channel):
    def __init__(self, is_3d=False):
        super().__init__('rough_temperature', 1, True)
        self.is_3d = is_3d

    def get(self, scan_results=None, timestep=None, *args, **kwargs):
        if self.is_3d:
            return [scan_results.rough_temperature_field[timestep]]

        return [scan_results.rough_temperature_field[timestep, :, :, -1]]


class RoughTemperatureDifference(Channel):
    def __init__(self, is_3d=False):
        super().__init__('rough_temperature_difference', 1, True)
        self.is_3d = is_3d

    def get(self, scan_results=None, timestep=None, *args, **kwargs):
        temperature = scan_results.rough_temperature_field[timestep]
        next_step_temperature = scan_results.rough_temperature_field[timestep + 1]

        next_step_diffferences = next_step_temperature - temperature

        if self.is_3d:
            return next_step_diffferences

        return [next_step_diffferences[:, :, -1]]


class RoughTemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=32, box_step_scale=0.25, return_coordinates=False):
        channel_count = 1 if not return_coordinates else 4 if is_3d else 3
        super().__init__('rough_temperature_around_laser', channel_count, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.box_step_scale = box_step_scale
        self.return_coordinates = return_coordinates

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        (x, y, z), t = scan_results.get_interpolated_grid_around_laser(timestep, self.box_size, self.box_step_scale, scan_parameters, False, self.is_3d)
        if self.return_coordinates:
            laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
            x = x - laser_x
            y = y - laser_y
            if self.is_3d:
                return [x, y, z, t]
            return [x, y, t]
        return [t]


class OffsetRoughTemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=24, box_step_scale=0.25, offset_ratio=2/3, return_coordinates=False):
        channel_count = 1 if not return_coordinates else 4 if is_3d else 3
        super().__init__('offset_rough_temperature_around_laser', channel_count, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.box_step_scale = box_step_scale
        self.return_coordinates = return_coordinates
        self.offset_ratio = offset_ratio

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        (x, y, z), t = scan_results.get_interpolated_grid_around_laser(timestep, self.box_size, self.box_step_scale, scan_parameters, False, self.is_3d, offset_ratio=self.offset_ratio)
        if self.return_coordinates:
            laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
            x = x - laser_x
            y = y - laser_y
            if self.is_3d:
                return [x, y, z, t]
            return [x, y, t]
        return [t]


class LowResRoughTemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=32, return_coordinates=False):
        channel_count = 1 if not return_coordinates else 4 if is_3d else 3
        super().__init__('low_res_rough_temperature_around_laser', channel_count, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.return_coordinates = return_coordinates

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        (x, y, z), t = scan_results.get_interpolated_grid_around_laser(timestep, self.box_size, 1, scan_parameters, False, self.is_3d)
        if self.return_coordinates:
            laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
            x = x - laser_x
            y = y - laser_y
            if self.is_3d:
                return [x, y, z, t]
            return [x, y, t]
        return [t]


class OffsetLowResRoughTemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=24, offset_ratio=2/3, return_coordinates=False):
        channel_count = 1 if not return_coordinates else 4 if is_3d else 3
        super().__init__('offset_low_res_rough_temperature_around_laser', channel_count, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.return_coordinates = return_coordinates
        self.offset_ratio = offset_ratio

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        (x, y, z), t = scan_results.get_interpolated_grid_around_laser(timestep, self.box_size, 1, scan_parameters, False, self.is_3d, offset_ratio=self.offset_ratio)
        if self.return_coordinates:
            laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
            x = x - laser_x
            y = y - laser_y
            if self.is_3d:
                return [x, y, z, t]
            return [x, y, t]
        return [t]


class TemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=32, box_step_scale=0.25):
        super().__init__('temperature_around_laser', 1, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.box_step_scale = box_step_scale

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        return [scan_results.get_interpolated_grid_around_laser(
            timestep, self.box_size, self.box_step_scale,
            scan_parameters, True, self.is_3d)[1]]


class OffsetTemperatureAroundLaser(Channel):
    def __init__(self, is_3d=False, box_size=24, box_step_scale=0.25, offset_ratio=2/3):
        super().__init__('offset_temperature_around_laser', 1, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.box_step_scale = box_step_scale
        self.offset_ratio = offset_ratio

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        return [scan_results.get_interpolated_grid_around_laser(
            timestep, self.box_size, self.box_step_scale,
            scan_parameters, True, self.is_3d, offset_ratio=self.offset_ratio)[1]]


class MeltPoolPointChunk(Channel):
    def __init__(self, is_3d=False, chunk_size=1024, input_shape=[32, 32], offset_ratio=None, include_gradients=False):
        channel_count = chunk_size
        super().__init__('melt_pool_point_chunk', channel_count, True)
        self.is_3d = is_3d
        self.chunk_size = chunk_size
        self.input_size = input_shape[0]
        self.offset_ratio = offset_ratio
        self.include_gradients = include_gradients
        self.point_len = 3
        if self.is_3d:
            self.point_len += 1
        if self.include_gradients:
            self.point_len += 3
        if self.is_3d and self.include_gradients:
            self.point_len += 1

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        if scan_results.is_melt_pool_empty(timestep):
            low_res_points = self.get_padding(scan_parameters, scan_results, timestep, self.chunk_size)
            chunks = self.subtract_laser_coordinates(scan_results, scan_parameters, timestep, [low_res_points])
            return chunks

        if self.include_gradients:
            high_res_points = scan_results.get_melt_pool_data_at_timestep(timestep, is_3d=self.is_3d)
        else:
            high_res_coordinates, high_res_temperatures = scan_results.get_melt_pool_coordinates_and_temperature(timestep, is_3d=self.is_3d)
            high_res_points = np.concatenate([high_res_coordinates, high_res_temperatures.reshape(-1, 1)], axis=1)

        chunk_count = (len(high_res_points) // self.chunk_size) + 1
        padding_length = self.chunk_size - (len(high_res_points) % self.chunk_size)

        chunk_sizes = [self.chunk_size] * (chunk_count - 1) + [self.chunk_size - padding_length]

        np.random.shuffle(high_res_points)

        chunks = []
        for size in chunk_sizes:
            chunks.append(high_res_points[:size])
            high_res_points = high_res_points[size:]

        padding = self.get_padding(scan_parameters, scan_results, timestep, padding_length)
        chunks[-1] = np.concatenate([chunks[-1], padding])

        chunks = self.subtract_laser_coordinates(scan_results, scan_parameters, timestep, chunks)

        return chunks

    def get_padding(self, scan_parameters, scan_results, timestep, padding_length):
        (x_coords, y_coords, z_coords), interpolated_values = scan_results.get_interpolated_grid_around_laser(
            timestep, size=self.input_size, step_scale=1,
            scan_parameters=scan_parameters, include_high_resolution_points=False, is_3d=self.is_3d, offset_ratio=self.offset_ratio)

        if self.is_3d:
            points = np.array(list(zip(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), interpolated_values.flatten())))
        else:
            points = np.array(list(zip(x_coords.flatten(), y_coords.flatten(), interpolated_values.flatten())))
        point_coordinates = points[:, :2]

        if point_coordinates.shape[0] == padding_length:
            if self.include_gradients:
                points = np.pad(points, constant_values=np.nan, mode='constant', pad_width=(0, self.point_len - padding.shape[-1]))
            return points

        # Prefer points closer to the laser
        laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
        weights = np.array([np.linalg.norm([laser_x - x, laser_y - y]) for x, y in point_coordinates])
        weights = 1 / weights
        probabilities = weights / np.sum(weights)

        indices = points.shape[0]
        indices = np.random.choice(indices, size=padding_length, replace=False, p=probabilities)
        padding = points[indices]

        if self.include_gradients:
            padding = np.pad(padding, constant_values=np.nan, mode='constant', pad_width=(0, self.point_len - padding.shape[-1]))

        return padding

    def subtract_laser_coordinates(self, scan_results, scan_parameters, timestep, chunks):
        laser_x, laser_y = scan_results.get_laser_coordinates_at_timestep(timestep)
        for chunk in chunks:
            chunk[:, 0] = chunk[:, 0] - laser_x
            chunk[:, 1] = chunk[:, 1] - laser_y
        return chunks


class LaserPositionInBox(Channel):
    def __init__(self, is_3d=False, box_size=32, box_step_scale=1, offset_ratio=None):
        super().__init__('laser_position_in_box', 1, True)
        self.is_3d = is_3d
        self.box_size = box_size
        self.box_step_scale = box_step_scale
        self.offset_ratio = offset_ratio

    def get(self, scan_parameters=None, scan_results=None, timestep=None, *args, **kwargs):
        laser_data = scan_results.get_laser_data_at_timestep(timestep)

        step_size = scan_parameters.rough_coordinates_step_size / self.box_step_scale

        if self.offset_ratio is not None:
            points = scan_parameters.get_offset_coordinates_around_position(
                laser_data, size=self.box_size, scale=self.box_step_scale, offset_ratio=self.offset_ratio, is_3d=self.is_3d)
        else:
            points = scan_parameters.get_coordinates_around_position(
                laser_data[0], laser_data[1], size=self.box_size, scale=self.box_step_scale, is_3d=self.is_3d)

        points = np.array(points).reshape((self.box_size, self.box_size, 16 // self.box_step_scale, 3))

        laser_grid = np.zeros((self.box_size, self.box_size, 16 // self.box_step_scale))

        laser_grid[:, :, -1] = np.min((1 - np.abs(points[:, :, -1, 0] - laser_data[0]) / step_size, 0)) * np.min((1 - np.abs(points[:, :, -1, 1] - laser_data[1]) / step_size, 0))

        return [laser_grid]


class ScanningAngle(Channel):
    def __init__(self, **kwargs):
        super().__init__('scanning_angle', 1, False)

    def get(self, scan_parameters=None, *args, **kwargs):
        return [scan_parameters.scanning_angle]


class LaserPower(Channel):
    def __init__(self, **kwargs):
        super().__init__('laser_power', 1, False)

    def get(self, scan_results=None, timestep=None, *args, **kwargs):
        return [scan_results.get_laser_power_at_timestep(timestep)]


class LaserRadius(Channel):
    def __init__(self, **kwargs):
        super().__init__('laser_radius', 1, False)

    def get(self, scan_results=None, timestep=None, *args, **kwargs):
        return [scan_results.get_laser_radius_at_timestep(timestep)]
