import numpy as np
from skimage import measure

from leap3d.config import MELTING_POINT


def get_melting_pool_contour(temperature_grid, threshold=MELTING_POINT):
    # Threshold the temperature grid to identify the melting pool region
    binary_grid = np.where(temperature_grid >= threshold, 1, 0)

    # Use marching cubes algorithm to extract the contour
    vertices, faces, _, _ = measure.marching_cubes(binary_grid, level=0, spacing=(1, 1, 1))

    return vertices, faces
