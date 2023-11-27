import functools
import logging
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
        projected_laser_x = project_point_to_cross_section(laser_x, laser_y, case_params.x_min, case_params.y_min)
        laser_position, = ax.plot([projected_laser_x, projected_laser_x], [case_params.z_min, case_params.z_max], 'red', lw=1)
        ims.append(laser_position)

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
    temperature = case_results.get_top_layer_temperatures(timestep)
    return plot_top_layer_temperature(ax, temperature, case_params, show_only_melt)


def plot_top_layer_temperature(ax, temperature, case_params, show_only_melt=False, vmax=None):
    X, Y = case_params.get_top_layer_coordinates()

    melting_point = case_params.melting_point
    if show_only_melt:
        temperature[temperature < melting_point] = 0
        temperature[temperature >= melting_point] = 1

    vmax = vmax if vmax is not None else 1 if show_only_melt else melting_point
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


def animate_cross_secion_and_top_player(case_results, case_params, frames=None):
    fig, (ax_0, ax_1) = plt.subplots(ncols=2)

    x_min, x_max, y_min, y_max, z_min, z_max = case_params.get_bounds()

    ax_0.set_xlim(
        -np.sqrt(x_min**2 + y_min**2),
        np.sqrt(x_max**2 + y_max**2))
    ax_0.set_ylim(z_min, z_max)

    frames = frames if frames is not None else case_results.total_timesteps
    animate_partial = functools.partial(plot_temperature_cross_section_at_timestep,
                                        ax=ax_0,
                                        case_results=case_results,
                                        case_params=case_params,
                                        show_scan_boundaries=True,
                                        show_laser_position=True,
                                        show_only_melt=False)
    ims_0 = [animate_partial(timestep=i) for i in range(0, frames, 5)]

    ax_1.set_xlim(x_min, x_max)
    ax_1.set_ylim(y_min, y_max)
    ax_1.set_aspect('equal', adjustable='box')

    frames = frames if frames is not None else case_results.total_timesteps
    ims_1 = []
    for i in range(0, frames, 5):
        ims_at_timestep = []

        im = plot_top_layer_temperature_at_timestep(ax_1, case_results, case_params, i, False)
        ims_at_timestep.append(im)

        im = plot_top_view_laser_position_at_timestep(ax_1, case_results, i)
        ims_at_timestep.append(im)

        ims_1.append(ims_at_timestep)

    plot_top_view_scan_boundaries(ax_1, case_params)

    fig.colorbar(ims_1[0][0], ax=ax_1)

    return fig, [ims_0_element + ims_1_element for ims_0_element, ims_1_element in zip(ims_0, ims_1)]


def plot_dataset_histograms(dataset):
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_xlim(xmin=-4, xmax=4)
    ax1.set_xlim(xmin=-4, xmax=4)

    z_scores_train, z_scores_target = dataset.get_temperature_z_scores()

    ax0.hist(z_scores_train, bins=50)
    ax0.set(title='Train Temperature Frequency Histogram', ylabel='Frequency')
    ax1.hist(z_scores_target, bins=50)
    ax1.set(title='Target Temperature Diff Frequency Histogram', ylabel='Frequency')
    return fig, (ax0, ax1)

def plot_model_top_layer_temperature_comparison_at_timestep(axes, case_params, model, input, target, extra_params=None):
    # input temp
    input_temp = input[0, -1, :, :]
    # input_temp = get untransformed
    im_0_0 = plot_top_layer_temperature(axes[0, 0], input_temp, case_params)
    # target temp diff and temp
    target_temp_diff = target[0, 0, :, :]
    # target_temp_diff = get untransformed
    im_0_1 = plot_top_layer_temperature(axes[0, 1], target_temp_diff, case_params)

    target_temp = input_temp + target_temp_diff
    im_1_1 = plot_top_layer_temperature(axes[1, 1], target_temp, case_params)

    # plot model output temp diff and temp
    model_output_diff = model(input, extra_params=extra_params)[0, 0, :, :]
    im_0_2 = plot_top_layer_temperature(axes[0, 2], model_output_diff, case_params)

    model_output_temp = input_temp + model_output_diff
    im_1_2 = plot_top_layer_temperature(axes[1, 2], model_output_temp, case_params)

    predicted_diff = model_output_diff - target_temp_diff
    im_1_0 = plot_top_layer_temperature(axes[1, 0], predicted_diff, case_params)

    return [im_0_0, im_0_1, im_0_2, im_1_0, im_1_1, im_1_2]


def plot_model_top_layer_temperature_comparison(case_params, model, dataset, steps=10, samples=100):
    fig, axes = plt.subplots(ncols=3, nrows=2)
    fig.set_size_inches(15, 10)
    axes[0, 0].set_title('Input Temperature')
    axes[0, 1].set_title('Target Temperature Diff')
    axes[0, 2].set_title('Model Output Temperature Diff')
    axes[1, 0].set_title('Predicted Temperature Diff')
    axes[1, 1].set_title('Target Temperature')
    axes[1, 2].set_title('Model Output Temperature')

    for ax in axes.flatten():
        ax.set_aspect('equal', adjustable='box')

    ims = []

    for sample_idx in range(0, len(dataset) - steps, (len(dataset) - steps) // samples):
        for i in range(sample_idx, sample_idx + steps):
            x_data, extra_params, _ = dataset[i]
            temperature_t0 = dataset.x_train[i][0, -1, :, :]
            temperature_diff_t1 = dataset.targets[i][0, 0, :, :]

            im_0_0 = plot_top_layer_temperature(axes[0, 0], temperature_t0, case_params)
            im_0_1 = plot_top_layer_temperature(axes[0, 1], temperature_diff_t1, case_params)
            temperature_t1 = temperature_t0 + temperature_diff_t1
            im_1_1 = plot_top_layer_temperature(axes[1, 1], temperature_t1, case_params)

            # plot model output temp diff and temp
            x_data = x_data.to(model.device)
            model_output_diff = model(x_data, extra_params=extra_params.to(model.device))[0, 0, :, :].to('cpu') * 10
            model_output_temp = np.add(temperature_t0, model_output_diff)

            im_0_2 = plot_top_layer_temperature(axes[0, 2], model_output_diff, case_params)
            im_1_2 = plot_top_layer_temperature(axes[1, 2], model_output_temp, case_params)

            predicted_diff = np.abs(model_output_diff - temperature_t1)
            im_1_0 = plot_top_layer_temperature(axes[1, 0], predicted_diff, case_params)

            ims.append([im_0_0, im_0_1, im_0_2, im_1_0, im_1_1, im_1_2])
    fig.colorbar(ims[0][0], ax=axes[0, 0])

    return fig, axes, ims
