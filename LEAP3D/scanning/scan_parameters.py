from pathlib import Path

import numpy as np

from LEAP3D.scanning import ScanningStrategy


class ScanParameters():
    def __init__(self, parameters_filename: str | Path, case_index: int,
                 x_min: float=None, x_max: float=None,
                 y_min: float=None, y_max: float=None,
                 z_min: float=None, z_max: float=None,
                 safety_offset: float=None,
                 melting_point: float=None):
        self.case_index = case_index
        self.params = np.load(parameters_filename)[case_index, :]
        self.substrate_temperature = self.params[2]
        self.scanning_angle = self.params[3]
        self.hatching_distance = self.params[4]
        self.scanning_strategy = ScanningStrategy.SERPENTINE if self.params[4] == 0 else ScanningStrategy.PARALLEL
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.safety_offset = safety_offset
        self.melting_point = melting_point

    def get_bounds(self):
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    def __repr__(self):
        return f"""Case index:\t{self.case_index}
Substrate Temperature:\t{self.substrate_temperature}
Scanning Angle (rad):\t{self.scanning_angle}
Hatching Distance (m):\t{self.hatching_distance}
Scanning Strategy:\t{self.scanning_strategy}"""
