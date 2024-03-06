from enum import Enum

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

