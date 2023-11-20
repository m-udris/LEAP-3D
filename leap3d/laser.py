import logging

import numpy as np


def unpack_laser_data(laser_data):
    laser_x, laser_y, laser_power, laser_beam_radius, laser_velocity_x, laser_velocity_y, is_melt_pool = laser_data
    return laser_x, laser_y, laser_power, laser_beam_radius, laser_velocity_x, laser_velocity_y, is_melt_pool


def get_next_laser_position(laser_data, case_params):
    laser_x, laser_y, _, _, laser_velocity_x, laser_velocity_y, _ = laser_data
    laser_x_min, laser_x_max, laser_y_min, laser_y_max = case_params.get_laser_bounds()
    timestep_duration = case_params.timestep_duration

    new_laser_x = laser_x
    if laser_velocity_x != 0:
        x_delta = timestep_duration * laser_velocity_x
        new_laser_x += x_delta.astype(np.float64)
        new_laser_x = np.clip(new_laser_x, laser_x_min, laser_x_max)

    new_laser_y = laser_y
    if laser_velocity_y != 0:
        y_delta = timestep_duration * laser_velocity_y
        new_laser_y += y_delta.astype(np.float64)
        new_laser_y = np.clip(new_laser_y, laser_y_min, laser_y_max)

    return new_laser_x, new_laser_y


def convert_laser_position_to_grid(laser_x, laser_y, rough_coordinates, is_3d=False):
    if is_3d:
        return convert_laser_position_to_grid_3d(laser_x, laser_y, rough_coordinates)
    return convert_laser_position_to_grid_2d(laser_x, laser_y, rough_coordinates)


def convert_laser_position_to_grid_2d(laser_x, laser_y, rough_coordinates):
    def get_overlap(laser_x, laser_y, x, y, x_step, y_step):
        """Get fraction of overlapping area of rectangles of size (x_step, y_step) centered at given points"""

        return (1 - abs(laser_x - x)/x_step) * (1 - abs(laser_y - y)/y_step)

    laser_grid = np.zeros_like(rough_coordinates['x_rough'][:, :, -1])

    # since the format of rough_coordinates is [x][y],
    # we extract a 1D array of unique increasing coordinate values
    x_rough = rough_coordinates['x_rough'][:, 0, -1]
    y_rough = rough_coordinates['y_rough'][0, :, -1]

    # get index of the first element that is >= laser_*
    x_index = np.argmax(x_rough >= laser_x)
    y_index = np.argmax(y_rough >= laser_y)

    x0 = x_rough[x_index]
    x1 = x_rough[x_index+1]
    y0 = y_rough[y_index]
    y1 = y_rough[y_index+1]

    x_step = x1 - x0
    y_step = y1 - y0

    laser_grid[x_index][y_index] = get_overlap(laser_x, laser_y, x0, y0, x_step, y_step)
    laser_grid[x_index][y_index+1] = get_overlap(laser_x, laser_y, x0, y1, x_step, y_step)
    laser_grid[x_index+1][y_index+1] = get_overlap(laser_x, laser_y, x1, y0, x_step, y_step)
    laser_grid[x_index+1][y_index+1] = get_overlap(laser_x, laser_y, x1, y1, x_step, y_step)

    return laser_grid

def convert_laser_position_to_grid_3d(laser_x, laser_y, rough_coordinates):
    top_layer_grid = convert_laser_position_to_grid_2d(laser_x, laser_y, rough_coordinates)
    grid = np.zeros_like(rough_coordinates['x_rough'])
    grid[:, :, -1] = top_layer_grid
    return grid