from matplotlib import pyplot as plt
import numpy as np
import scipy
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from leap3d.config import MELTING_POINT, DATA_DIR, DATASET_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH
from leap3d.scanning import ScanParameters, ScanResults


def get_melting_pool_contour_3d(temperature_grid, threshold=MELTING_POINT, top_k=None):
    mask = None
    if top_k is not None:
        mask = np.zeros_like(temperature_grid, dtype=bool)
        mask[:, :, -top_k:] = True
    # Use marching cubes algorithm to extract the contour
    try:
        vertices, faces, _, _ = measure.marching_cubes(temperature_grid, level=threshold, mask=mask)
    except (ValueError, RuntimeError):
        return [], []

    return vertices, faces


def get_melting_pool_contour_2d(temperature_grid, threshold=MELTING_POINT, top_k=None):
    """Get contours of top_k layers of the temperature grid. The output is a list of contours for each layer, starting from the top layer.
    """
    # Threshold the temperature grid to identify the melting pool region
    contours = []

    top_k = top_k or temperature_grid.shape[2]
    for i in range(1, min(top_k, temperature_grid.shape[2])):
        current_layer = temperature_grid[:, :, -i]
        contours.append(measure.find_contours(current_layer, level=threshold))

    return contours


if __name__ == '__main__':
    scan_results = ScanResults(DATA_DIR / 'case_0000.npz')
    timestep = 2000
    scan_parameters = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=1)
    coordinates, temperature = scan_results.get_coordinate_points_and_temperature_at_timestep(scan_parameters, timestep, resolution='high')

    xi = np.linspace(scan_parameters.x_min, scan_parameters.x_max, 256)
    yi = np.linspace(scan_parameters.y_min, scan_parameters.y_max, 256)
    zi = np.linspace(scan_parameters.z_min, scan_parameters.z_max, 64)

    new_coordinate_points = np.array([(x, y, z) for x in xi for y in yi for z in zi])
    temperature = scipy.interpolate.griddata(coordinates, temperature, new_coordinate_points, method='nearest')
    temperature = temperature.reshape((256, 256, 64))
    print(temperature.shape)

    vertices, faces = get_melting_pool_contour_3d(temperature)
    print(vertices.shape)
    print(faces.shape)
    print(vertices)
    print(faces)

    ax = plt.figure().add_subplot(projection='3d')

    for vertice in vertices:
        ax.plot(vertices[:, 1], vertices[:, 0], vertices[:, 2], linewidth=2)

    ax.set_xlim(0, temperature.shape[0])
    ax.set_ylim(temperature.shape[1], 0)
    ax.set_zlim(0, temperature.shape[2])
    ax.set_title('3D Contour Plot - vertices')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mesh = Poly3DCollection(vertices[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, temperature.shape[0])
    ax.set_ylim(temperature.shape[1], 0)
    ax.set_zlim(0, temperature.shape[2])
    ax.set_title('3D Contour Plot - faces')
    plt.show()


    ax = plt.figure().add_subplot(projection='3d')

    for depth in range(temperature.shape[2]):
        print(depth)
        top_layer_temperature = temperature[:, :, depth]
        contours = get_melting_pool_contour_2d(top_layer_temperature)
        print(contours)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], depth, linewidth=2)

    ax.set_xlim(0, temperature.shape[0])
    ax.set_ylim(temperature.shape[1], 0)
    ax.set_zlim(0, temperature.shape[2])
    ax.set_title('3D Contour Plot - Layered 2D Contours')
    plt.show()

    top_layer_temperature = temperature[:, :, -1]
    contours = get_melting_pool_contour_2d(top_layer_temperature)
    print(contours)

    fig, ax = plt.subplots()
    ax.imshow(top_layer_temperature)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_title('2D Contour Plot - Top view')

    plt.show()
