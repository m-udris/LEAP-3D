import functools
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

def project_point_to_cross_section(x, y, x_min, y_min):
        return np.sqrt((x - x_min)**2 + (y - y_min)**2) - np.sqrt(x_min**2 + y_min**2)


def draw_cross_section_scanning_bounds(ax, case_params, laser_x, laser_y):
    x_min, _, y_min, _, z_min, z_max = case_params.get_bounds()

    laser_x_min, laser_x_max, laser_y_min, laser_y_max, *_ = case_params.get_cross_section_scanning_bounds(laser_x, laser_y)

    # Project the scanning bounds into the cross section
    cs_x_min = project_point_to_cross_section(laser_x_min, laser_y_min, x_min, y_min)
    cs_x_max = project_point_to_cross_section(laser_x_max, laser_y_max, x_min, y_min)

    # Plot the scanning bounds
    scanning_bound_left, = ax.plot([cs_x_min, cs_x_min], [z_min, z_max], 'black', lw=1)
    scanning_bound_right, = ax.plot([cs_x_max, cs_x_max], [z_min, z_max], 'black', lw=1)

    return scanning_bound_left, scanning_bound_right


def plot_cross_section_along_laser_at_timestep(ax, case_results, case_params, timestep,
                                               interpolation_steps=128, interpolation_method='nearest', show_only_melt=False):
    T_interpolated, xi, yi, zi = case_results.get_interpolated_data_along_laser_at_timestep(
        case_params, timestep,
        return_grid=True, steps=interpolation_steps, method=interpolation_method)

    melting_point = case_params.melting_point

    if show_only_melt:
        T_interpolated[T_interpolated < melting_point] = 0
        T_interpolated[T_interpolated >= melting_point] = 1

    # Project x and y axes into 1D
    new_x = project_point_to_cross_section(xi, yi, case_params.x_min, case_params.y_min)

    # Reshape coordinates for plotting
    new_x_grid = [[x] * zi.shape[0] for x in new_x]
    new_z_grid = [zi for _ in new_x]
    T_values = T_interpolated.reshape((new_x.shape[0], zi.shape[0]))

    vmax = 1 if show_only_melt else melting_point
    im = ax.pcolormesh(new_x_grid, new_z_grid, T_values, animated=True, vmax=vmax)

    return im


def plot_temperature_cross_section_at_timestep(ax, case_results, case_params, timestep, interpolation_steps=96, show_scan_boundaries=True, show_laser_position=False, show_only_melt=False):
        # TODO: add laser position
        ims = []

        laser_x, laser_y = case_results.get_laser_coordinates_at_timestep(timestep)

        cross_section_plot = plot_cross_section_along_laser_at_timestep(
            ax, case_results, case_params, timestep, interpolation_steps, show_only_melt=show_only_melt)

        ims.append(cross_section_plot)


        if show_scan_boundaries:
            scanning_bound_left, scanning_bound_right = draw_cross_section_scanning_bounds(
                ax, case_params, laser_x, laser_y)
            ims.append(scanning_bound_left)
            ims.append(scanning_bound_right)

        if show_laser_position:
            raise NotImplementedError()

        return ims


def get_frames_for_temperature_cross_section_animation(case_results, case_params,
                                      frames=None, show_scan_boundaries=True,
                                      show_laser_position=False, show_only_melt=False):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    ax.set_xlim(
        -np.sqrt(case_params.x_min**2 + case_params.y_min**2),
        np.sqrt(case_params.x_max**2 + case_params.y_max**2))

    frames = frames if frames is not None else case_results.total_timesteps
    animate_partial = functools.partial(plot_temperature_cross_section_at_timestep,
                                        ax=ax,
                                        case_results=case_results,
                                        case_params=case_params,
                                        show_scan_boundaries=show_scan_boundaries,
                                        show_laser_position=show_laser_position,
                                        show_only_melt=show_only_melt)
    ims = [animate_partial(timestep=i) for i in range(0, frames, 5)]

    if not show_only_melt:
        fig.colorbar(ims[0][0], ax=ax)

    return fig, ims


def plot_top_layer_temperature_at_timestep(ax, case_results, case_params, timestep, show_only_melt=False):
        X, Y, temperature = case_results.get_top_layer_temperatures(timestep)

        melting_point = case_params.melting_point

        if show_only_melt:
            temperature[temperature < melting_point] = 0
            temperature[temperature >= melting_point] = 1

        vmax = 1 if show_only_melt else melting_point
        im = ax.pcolormesh(X, Y, temperature, animated=True, vmax=vmax)
        return im


def plot_top_view_scan_boundaries(ax, case_params):
    laser_x_min, laser_x_max, laser_y_min, laser_y_max = case_params.get_laser_bounds()
    ax.plot([laser_x_min, laser_x_min], [laser_y_min, laser_y_max], 'black', lw=1)
    ax.plot([laser_x_max, laser_x_max], [laser_y_min, laser_y_max], 'black', lw=1)
    ax.plot([laser_x_min, laser_x_max], [laser_y_min, laser_y_min], 'black', lw=1)
    ax.plot([laser_x_min, laser_x_max], [laser_y_max, laser_y_max], 'black', lw=1)


def plot_top_view_laser_position_at_timestep(ax, case_results, timestep):
    laser_x, laser_y = case_results.get_laser_coordinates_at_timestep(timestep)
    im = ax.scatter(laser_x, laser_y, animated=True, c='red')
    return im


def get_frames_for_top_layer_temperatures(case_results, case_params, frames=None,
                                          show_scan_boundaries=True, show_laser_position=True,
                                          show_only_melt=False):
    x_min, x_max, y_min, y_max, *_ = case_params.get_bounds()

    fig, ax = plt.subplots(sharex=True, sharey=True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    frames = frames if frames is not None else case_results.total_timesteps
    ims = []
    for i in range(0, frames, 5):
        ims_at_timestep = []

        im = plot_top_layer_temperature_at_timestep(ax, case_results, case_params, i, show_only_melt)
        ims_at_timestep.append(im)

        if show_laser_position:
            im = plot_top_view_laser_position_at_timestep(ax, case_results, i)
            ims_at_timestep.append(im)

        ims.append(ims_at_timestep)

    if show_scan_boundaries:
        plot_top_view_scan_boundaries(ax, case_params)

    if not show_only_melt:
        fig.colorbar(ims[0][0], ax=ax)

    return fig, ims