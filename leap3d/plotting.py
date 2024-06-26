import functools
import logging
from matplotlib import animation, pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import interpolate
import torch
import skimage
import trimesh

from leap3d.contour import get_melting_pool_contour_2d, get_melting_pool_contour_3d

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


def get_projected_cross_section(case_results, case_params, timestep, interpolation_steps=128, interpolation_method='nearest', show_only_melt=False):
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
    return new_x_grid, new_z_grid, T_values


def plot_cross_section_along_laser_at_timestep(ax, case_results, case_params, timestep,
                                               interpolation_steps=128, interpolation_method='nearest', show_only_melt=False, use_laser_position: tuple[float, float] = None):
    T_interpolated, xi, yi, zi = case_results.get_interpolated_data_along_laser_at_timestep(
        case_params, timestep,
        return_grid=True, steps=interpolation_steps, method=interpolation_method, use_laser_position=use_laser_position)

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
    im = ax.pcolormesh(new_x_grid, new_z_grid, T_values, animated=True, vmax=vmax, cmap='hot')

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


def plot_top_layer_temperature_at_timestep(ax, case_results, case_params, timestep, show_only_melt=False, cmap=None):
    temperature = case_results.get_top_layer_temperatures(timestep)
    return plot_top_layer_temperature(ax, temperature, case_params, show_only_melt, cmap=cmap)


def plot_top_layer_temperature(ax, temperature, case_params, show_only_melt=False, vmin=None, vmax=None, cmap=None, animated=False):
    X, Y = case_params.get_top_layer_coordinates()

    if len(temperature.shape) == 3:
        temperature = temperature[:, :, -1]

    melting_point = case_params.melting_point
    if show_only_melt:
        temperature[temperature < melting_point] = 0
        temperature[temperature >= melting_point] = 1

    vmax = vmax if vmax is not None else 1 if show_only_melt else melting_point
    im = ax.pcolormesh(X, Y, temperature, animated=animated, vmin=vmin, vmax=vmax, cmap=cmap)
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
    axes[0, 0].set_title('GT Temperature Diff')
    axes[0, 1].set_title('P Temperature Diff')
    axes[0, 2].set_title('Temperature Diff Error')
    axes[1, 0].set_title('GT Temperature')
    axes[1, 1].set_title('P Temperature')
    axes[1, 2].set_title('Temperature Error')

    for ax in axes.flatten():
        ax.set_aspect('equal', adjustable='box')

    ims = []

    for sample_idx in range(0, len(dataset) - steps, (len(dataset) - steps) // samples):
        temperature_p = None
        for i in range(sample_idx, sample_idx + steps):
            print(i, end='\r')
            x_data, extra_params, _ = dataset[i]
            starting_temperature = dataset.inputs[i][0, -1]
            temperature_diff_gt = dataset.targets[i][0, 0]

            temperature_gt = starting_temperature + temperature_diff_gt
            # plot model output temp diff and temp
            model_input = x_data.to(model.device)

            if temperature_p is None:
                temperature_p = starting_temperature
            model_input[0, -1] = dataset.input_transform(torch.tensor(temperature_p).clone().to(model.device))
            temperature_diff_p_normalized = model(model_input, extra_params=extra_params.to(model.device))[0, 0]
            temperature_diff_p = dataset.target_inverse_transform(temperature_diff_p_normalized.to('cpu')).numpy()

            temperature_diff_error = -(temperature_diff_gt - temperature_diff_p)
            temperature_error = -(temperature_gt - temperature_p)

            im_0_0 = plot_top_layer_temperature(axes[0, 0], temperature_diff_gt, case_params, vmin=-1000, vmax=1000, cmap='seismic')
            im_0_1 = plot_top_layer_temperature(axes[0, 1], temperature_diff_p, case_params, vmin=-1000, vmax=1000, cmap='seismic')
            im_0_2 = plot_top_layer_temperature(axes[0, 2], temperature_diff_error, case_params, vmin=-1000, vmax=1000, cmap='seismic')

            im_1_0 = plot_top_layer_temperature(axes[1, 0], temperature_gt, case_params, vmin=0, vmax=3000, cmap=None)
            im_1_1 = plot_top_layer_temperature(axes[1, 1], temperature_p, case_params, vmin=0, vmax=3000, cmap=None)
            im_1_2 = plot_top_layer_temperature(axes[1, 2], temperature_error, case_params, vmin=0, vmax=3000, cmap='seismic')

            temperature_p = np.add(temperature_p, temperature_diff_p)

            ims.append([im_0_0, im_0_1, im_0_2, im_1_0, im_1_1, im_1_2])
    fig.colorbar(ims[0][0], ax=axes[0, 0])
    fig.colorbar(ims[0][1], ax=axes[0, 1])
    fig.colorbar(ims[0][2], ax=axes[0, 2])
    fig.colorbar(ims[0][3], ax=axes[1, 0])
    fig.colorbar(ims[0][4], ax=axes[1, 1])
    fig.colorbar(ims[0][5], ax=axes[1, 2])

    return fig, axes, ims


def plot_2d_contours(contours, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ims = []
    for contour in contours:
        ims += ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    return ims


def plot_fake_3d_contours(contours, max_n_layers=64, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ims = []
    for (depth, contour_layer) in enumerate(contours):
        for contour in contour_layer:
            ims += ax.plot(contour[:, 0], contour[:, 1], max_n_layers - depth - 1, linewidth=2)
    return ims


def plot_3d_contours_vertices(vertices, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    im, = ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', linewidth=2)
    return [im]


def plot_3d_contours_faces(vertices, faces, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if len(ax.collections) == 0:
        mesh = Poly3DCollection(vertices[faces], linewidths=1, edgecolors='k')
        ax.add_collection3d(mesh)
    else:
        ax.collections[0].set_verts(vertices[faces])
        return [ax.collections[0]]
    return [mesh]


def get_contour_animation(scan_results, scan_parameters, t_start=0, t_end=None, t_stride=5, denoise=None, animation_filepath="./plots/contours.gif", **kwargs):
    def get_ims(timestep):
        for ax in axes[:-1]:
            ax.clear()
        ims_at_timestep = []
        coordinates, temperature = scan_results.get_interpolated_data(scan_parameters, timestep)
        laser_coordinates = scan_results.get_laser_coordinates_at_timestep(timestep)
        x_min = min(coordinates, key=lambda x: x[0])[0]
        y_min = min(coordinates, key=lambda x: x[1])[1]
        x_max = max(coordinates, key=lambda x: x[0])[0]
        y_max = max(coordinates, key=lambda x: x[1])[1]
        x_step_size = (x_max - x_min) / 256
        y_step_size = (y_max - y_min) / 256
        laser_x = (laser_coordinates[0] - x_min) / x_step_size
        laser_y = (laser_coordinates[1] - y_min) / y_step_size

        if denoise == 'wavelet':
            temperature = 255 * temperature / 1500
            temperature = skimage.restoration.denoise_wavelet(temperature, **kwargs)
            temperature = temperature * 1500 / 255

        contours_2d = get_melting_pool_contour_2d(temperature, top_k=24)

        im = axes[0].imshow(temperature[:, :, -1], vmin=300, vmax=1500, cmap='hot', animated=True)
        ims_at_timestep.append(im)

        ims_at_timestep += plot_2d_contours(contours_2d[0], axes[0])

        ims_at_timestep += plot_fake_3d_contours(contours_2d, max_n_layers=temperature.shape[2], ax=axes[1])

        vertices_3d, faces_3d = get_melting_pool_contour_3d(temperature, top_k=24)

        if len(vertices_3d) == 0:
            axes[2].clear()
            axes[3].clear()
            return ims_at_timestep

        if denoise == 'laplace':
            mesh = trimesh.Trimesh(vertices_3d, faces_3d)
            trimesh.smoothing.filter_laplacian(mesh)
            vertices_3d = mesh.vertices
            faces_3d = mesh.faces
        if denoise == 'humphrey':
            mesh = trimesh.Trimesh(vertices_3d, faces_3d)
            trimesh.smoothing.filter_humphrey(mesh, kwargs.get('beta', 0.5))
            vertices_3d = mesh.vertices
            faces_3d = mesh.faces

        ims_at_timestep += plot_3d_contours_vertices(vertices_3d, ax=axes[2])

        plot_3d_contours_faces(vertices_3d, faces_3d, ax=axes[3])

        for ax in axes[1:]:
            ax.set_xlim(laser_x - 15, laser_x + 15)
            ax.set_ylim(laser_y - 15, laser_y + 15)
            ax.set_zlim(40, 64)

        return ims_at_timestep

    fig = plt.figure(figsize=(24, 10))
    axes = []
    axes.append(fig.add_subplot(1, 4, 1))
    axes.append(fig.add_subplot(1, 4, 2, projection='3d'))
    axes.append(fig.add_subplot(1, 4, 3, projection='3d'))
    axes.append(fig.add_subplot(1, 4, 4, projection='3d'))

    for ax in axes[1:]:
        ax.set_zlim(40, 64)

    frames = ((t_end or scan_results.timesteps) - t_start) // t_stride
    ani = animation.FuncAnimation(fig, lambda t: get_ims(t_start + t*t_stride), frames=frames, repeat=True, interval=500, blit=True, repeat_delay=1000)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(axes[0].get_images()[0], cax=cax, orientation='vertical')
    ani.save(animation_filepath, fps=1, progress_callback=print)
