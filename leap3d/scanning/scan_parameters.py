from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing

from leap3d.config import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SAFETY_OFFSET, MELTING_POINT, TIMESTEP_DURATION, SCALE_DISTANCE_BY, SCALE_TIME_BY
from leap3d.scanning import ScanningStrategy

class ScanParameters():
    def __init__(self, parameters: numpy.typing.NDArray | str | Path,
                 rough_coordinates: numpy.typing.NDArray | str | Path, case_index: int,
                 x_min: float=None, x_max: float=None,
                 y_min: float=None, y_max: float=None,
                 z_min: float=None, z_max: float=None,
                 safety_offset: float=None,
                 melting_point: float=None,
                 timestep_duration: float=None,
                 scale_distance_by: float=None,
                 scale_time_by: float=None):
        self.case_index = case_index

        self.scale_distance_by = scale_distance_by or SCALE_DISTANCE_BY
        self.scale_time_by = scale_time_by or SCALE_TIME_BY

        if isinstance(parameters, str) or isinstance(parameters, Path):
            self.params = np.load(parameters)[case_index, :]
        else:
            self.params = parameters[case_index, :]

        if isinstance(rough_coordinates, str) or isinstance(rough_coordinates, Path):
            self.rough_coordinates = dict(np.load(rough_coordinates))
        else:
            self.rough_coordinates = dict(rough_coordinates)
        self.rough_coordinates['x_rough'] *= self.scale_distance_by
        self.rough_coordinates['y_rough'] *= self.scale_distance_by
        self.rough_coordinates['z_rough'] *= self.scale_distance_by

        self.substrate_temperature = self.params[2]
        self.scanning_angle = self.params[3]
        self.hatching_distance = self.params[4] * self.scale_distance_by
        self.scanning_strategy = ScanningStrategy.SERPENTINE if self.params[4] == 0 else ScanningStrategy.PARALLEL
        self.melting_point = melting_point or MELTING_POINT
        self.timestep_duration = (timestep_duration or TIMESTEP_DURATION) * self.scale_time_by

        self.x_min = (x_min or X_MIN) * self.scale_distance_by
        self.x_max = (x_max or X_MAX) * self.scale_distance_by
        self.y_min = (y_min or Y_MIN) * self.scale_distance_by
        self.y_max = (y_max or Y_MAX) * self.scale_distance_by
        self.z_min = (z_min or Z_MIN) * self.scale_distance_by
        self.z_max = (z_max or Z_MAX) * self.scale_distance_by
        self.safety_offset = (safety_offset or SAFETY_OFFSET) * self.scale_distance_by
        self.laser_x_min = (self.x_min + self.safety_offset)
        self.laser_x_max = (self.x_max - self.safety_offset)
        self.laser_y_min = (self.y_min + self.safety_offset)
        self.laser_y_max = (self.y_max - self.safety_offset)

        self.rough_coordinates_step_size = self.get_rough_coordinates_step_size()

    def get_bounds(self):
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    def get_laser_bounds(self):
        return self.laser_x_min, self.laser_x_max, \
               self.laser_y_min, self.laser_y_max

    def get_rough_coordinates_step_size(self):
        return np.abs(self.rough_coordinates['x_rough'][1, 0, 0] - self.rough_coordinates['x_rough'][0, 0, 0])

    def get_cross_section_scanning_bounds(self, laser_x, laser_y):
        """Calculate scanning bounds of the cross section"""
        laser_angle_tan = np.tan(self.scanning_angle)

        cs_laser_y_min = self.laser_y_min
        cs_laser_x_min = (cs_laser_y_min - laser_y) / laser_angle_tan + laser_x

        # Check if new value is out of bounds and recalculate if needed
        if cs_laser_x_min < self.laser_x_min:
            cs_laser_x_min = self.laser_x_min
            cs_laser_y_min = laser_angle_tan * (cs_laser_x_min - laser_x) + laser_y

        cs_laser_y_max = self.laser_y_max
        cs_laser_x_max = (cs_laser_y_max - laser_y) / laser_angle_tan + laser_x

        # Check if new value is out of bounds and recalculate if needed
        if cs_laser_x_max > self.laser_x_max:
            cs_laser_x_max = self.laser_x_max
            cs_laser_y_max = laser_angle_tan * (cs_laser_x_max - laser_x) + laser_y

        return cs_laser_x_min, cs_laser_x_max, cs_laser_y_min, cs_laser_y_max

    def get_top_layer_coordinates(self):
        x_coordinates = self.rough_coordinates['x_rough'][:, :, -1]
        y_coordinates = self.rough_coordinates['y_rough'][:, :, -1]

        return x_coordinates, y_coordinates

    def get_rough_coordinates_points_list(self):
        X_rough = self.rough_coordinates['x_rough']
        Y_rough = self.rough_coordinates['y_rough']
        Z_rough = self.rough_coordinates['z_rough']

        points = []
        for X_planes, Y_planes, Z_planes in zip(X_rough, Y_rough, Z_rough):
            for X_rows, Y_rows, Z_rows in zip(X_planes, Y_planes, Z_planes):
                for x, y, z in zip(X_rows, Y_rows, Z_rows):
                    points.append((x, y, z))
        return points

    def get_bounds_around_position(self, x: float, y: float, size: int, depth=None):
        step_size = self.rough_coordinates_step_size
        x_start = x - (step_size * 0.5 * (size))
        x_end = x + (step_size * 0.5 * (size))
        y_start = y - (step_size * 0.5 * (size))
        y_end = y + (step_size * 0.5 * (size))

        if depth is not None:
            z_start = self.rough_coordinates['z_rough'][0, 0, -depth]
        else:
            z_start = self.rough_coordinates['z_rough'][0, 0, 0]
        z_end = self.rough_coordinates['z_rough'][0, 0, -1]

        return x_start, x_end, y_start, y_end, z_start, z_end

    def get_coordinates_around_position(self, x: float, y: float, size: int, scale: float=0.25, is_3d: bool=False, depth=None):
        bounds = self.get_bounds_around_position(x, y, size, depth=depth)

        return self.get_coordinates_within_bounds(bounds, size, scale, is_3d, depth=depth)

    def get_coordinates_within_bounds(self,
                                      bounds: Tuple[float, float, float, float, float, float],
                                      size: int, scale: float=0.25, is_3d: bool=False, depth=None):
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        x_points = np.linspace(x_min, x_max, int(size // scale))
        y_points = np.linspace(y_min, y_max, int(size // scale))

        if is_3d:
            z_depth = depth or len(self.rough_coordinates['z_rough'][0, 0, :])
            z_points = np.linspace(z_min, z_max, int(z_depth // scale))
        else:
            z_points = [np.max(self.rough_coordinates['z_rough'][0, 0, :])]

        points = [(x, y, z) for x in x_points for y in y_points for z in z_points]

        return points

    def get_offset_coordinates_around_position(self, laser_data, size: int=16, scale: float=0.25, offset_ratio: float=0.5, is_3d: bool=False, depth=None):
        bounds = self.get_offset_bounds_around_laser(laser_data, size, offset_ratio, depth=depth)

        return self.get_coordinates_within_bounds(bounds, size, scale, is_3d, depth=depth)

    def get_offset_bounds_around_laser(self, laser_data, size: int=16, offset_ratio: float=0.5, depth=None):
        laser_x, laser_y = laser_data[:2]
        angle = self.scanning_angle
        step_size = self.rough_coordinates_step_size
        laser_velocity_x = laser_data[4]
        direction = -np.sign(laser_velocity_x)

        radius = size * step_size / 2
        center_x = laser_x + offset_ratio * direction * radius * np.cos(angle)
        center_y = laser_y + offset_ratio * direction * radius * np.sin(angle)

        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounds_around_position(center_x, center_y, size, depth)

        scan_x_min = self.laser_x_min
        scan_x_max = self.laser_x_max
        scan_y_min = self.laser_y_min
        scan_y_max = self.laser_y_max

        oob_limit = radius * (1 - offset_ratio)
        if x_min < scan_x_min - oob_limit:
            x_max += scan_x_min - oob_limit - x_min
            x_min = scan_x_min - oob_limit
        if x_max > scan_x_max + oob_limit:
            x_min -= x_max - scan_x_max - oob_limit
            x_max = scan_x_max + oob_limit
        if y_min < scan_y_min - oob_limit:
            y_max += scan_y_min - oob_limit - y_min
            y_min = scan_y_min - oob_limit
        if y_max > scan_y_max + oob_limit:
            y_min -= y_max - scan_y_max - oob_limit
            y_max = scan_y_max + oob_limit

        return x_min, x_max, y_min, y_max, z_min, z_max

    def sample_coordinates_within_bounds(self, bounds: Tuple[float, float, float, float, float, float], sample_size: int, is_3d: bool=False):
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        x_points = np.random.random_sample(sample_size) * (x_max - x_min) + x_min
        y_points = np.random.random_sample(sample_size) * (y_max - y_min) + y_min

        if is_3d:
            z_points = np.random.random_sample(sample_size) * (z_max - z_min) + z_min
        else:
            z_points = [np.max(self.rough_coordinates['z_rough'][0, 0, :])] * sample_size

        points = list(zip(x_points, y_points, z_points))

        return points

    def __repr__(self):
        return f"""Case index:\t{self.case_index}
Substrate Temperature:\t{self.substrate_temperature}1
Scanning Angle (rad):\t{self.scanning_angle}
Hatching Distance (m):\t{self.hatching_distance}
Scanning Strategy:\t{self.scanning_strategy}"""
