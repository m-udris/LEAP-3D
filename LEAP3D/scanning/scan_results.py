from pathlib import Path

import numpy as np


class ScanResults():
    def __init__(self, results_filename: str | Path, rough_coordinates_filename: str | Path):
        self.results = np.load(results_filename, allow_pickle=True)
        self.rough_temperature_field = self.results['Rough_T_field']
        self.melt_pool = self.results['Melt_pool']
        self.laser_information = self.results['Laser_information']
        self.rough_coordinates = np.load(rough_coordinates_filename)
        self.total_timesteps = self.rough_temperature_field.shape[0]

    def get_rough_temperatures_at_timestep(self, timestep):
        return self.rough_temperature_field[timestep]

    def get_top_layer_temperatures(self, timestep):
        top_layer_results_at_timestep = self.rough_temperature_field[timestep, :, :, -1]
        X_rough = self.rough_coordinates['x_rough'][:, :, -1]
        Y_rough = self.rough_coordinates['y_rough'][:, :, -1]

        return X_rough, Y_rough, top_layer_results_at_timestep

        # TODO: Fix
        melt_pool_data = self.melt_pool[timestep]


        if len(melt_pool_data[0]) == 0:
            return X_rough, Y_rough, top_layer_results_at_timestep

        melt_pool_z_max = max(melt_pool_data, key=lambda x: x[2])[2]
        melt_pool_data = filter(lambda x: x[2] == melt_pool_z_max, melt_pool_data)
        melt_pool_data = sorted(melt_pool_data, key=lambda x: x[:2])

        X_melt = []
        Y_melt = []
        temperature_melt = []
        previous_x = None
        x_row = []
        y_row = []
        for (x, y, _, temperature, *_) in melt_pool_data:
            if previous_x is not None and previous_x != x:
                X_melt.append(x_row)
                Y_melt.append(y_row)
                x_row = []
                y_row = []
            x_row.append(x)
            y_row.append(y)
            temperature_melt.append(temperature)
            previous_x = x

        X = X_rough + X_melt
        Y = Y_rough + Y_melt
        T = top_layer_results_at_timestep + temperature_melt

        return X, Y, T

    def get_rough_coordinates_for_cross_section(self):
        X_rough = self.rough_coordinates['x_rough']
        Y_rough = self.rough_coordinates['y_rough']
        Z_rough = self.rough_coordinates['z_rough']

        points = []
        for X_planes, Y_planes, Z_planes in zip(X_rough, Y_rough, Z_rough):
            for X_rows, Y_rows, Z_rows in zip(X_planes, Y_planes, Z_planes):
                for x, y, z in zip(X_rows, Y_rows, Z_rows):
                    points.append((x, y, z))
        return points

    def get_laser_coordinates_at_timestep(self, timestep: int) -> [float, float]:
        """Get laser x and y coordinates at timestep"""
        laser_information = self.laser_information[timestep]
        return laser_information[0], laser_information[1], laser_information[2]

    def get_melt_pool_coordinates_and_temperature(self, timestep):
        coordinates = [point[:3] for point in self.melt_pool[timestep] if len(point) != 0]
        temperatures = [point[3] for point in self.melt_pool[timestep] if len(point) != 0]

        return coordinates, temperatures
