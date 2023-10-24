from pathlib import Path

import numpy as np

from leap3d.scanning import ScanningStrategy


class ScanParameters():
    def __init__(self, parameters_filename: str | Path, case_index: int,
                 x_min: float=None, x_max: float=None,
                 y_min: float=None, y_max: float=None,
                 z_min: float=None, z_max: float=None,
                 safety_offset: float=None,
                 melting_point: float=None,
                 timestep_duration: float=None):
        self.case_index = case_index
        self.params = np.load(parameters_filename)[case_index, :]
        self.substrate_temperature = self.params[2]
        self.scanning_angle = self.params[3]
        self.hatching_distance = self.params[4]
        self.scanning_strategy = ScanningStrategy.SERPENTINE if self.params[4] == 0 else ScanningStrategy.PARALLEL
        self.melting_point = melting_point
        self.timestep_duration = timestep_duration

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.safety_offset = safety_offset
        self.laser_x_min = x_min + safety_offset
        self.laser_x_max = x_max - safety_offset
        self.laser_y_min = y_min + safety_offset
        self.laser_y_max = y_max - safety_offset

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

    def __repr__(self):
        return f"""Case index:\t{self.case_index}
Substrate Temperature:\t{self.substrate_temperature}
Scanning Angle (rad):\t{self.scanning_angle}
Hatching Distance (m):\t{self.hatching_distance}
Scanning Strategy:\t{self.scanning_strategy}"""
