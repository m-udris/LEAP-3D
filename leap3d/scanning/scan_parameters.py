from pathlib import Path

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

    def get_bounds(self):
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    def get_laser_bounds(self):
        return self.laser_x_min, self.laser_x_max, \
               self.laser_y_min, self.laser_y_max

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

    def __repr__(self):
        return f"""Case index:\t{self.case_index}
Substrate Temperature:\t{self.substrate_temperature}
Scanning Angle (rad):\t{self.scanning_angle}
Hatching Distance (m):\t{self.hatching_distance}
Scanning Strategy:\t{self.scanning_strategy}"""
