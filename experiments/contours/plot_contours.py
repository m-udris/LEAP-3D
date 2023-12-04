import sys
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

from leap3d.contour import get_melting_pool_contour_2d, get_melting_pool_contour_3d
from leap3d.plotting import get_contour_animation
from leap3d.scanning.scan_parameters import ScanParameters
from leap3d.scanning.scan_results import ScanResults
from leap3d.config import DATA_DIR, PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH


if __name__ == '__main__':
    scan_results = ScanResults(DATA_DIR / 'case_0000.npz')
    t_start = 2000
    t_end = 3000
    t_stride = 5
    scan_parameters = ScanParameters(PARAMS_FILEPATH, ROUGH_COORDS_FILEPATH, case_index=0)

    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise=None, animation_filepath='./plots/contour_animation.gif')
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='wavelet', animation_filepath='./plots/contour_animation_wavelet_01.gif', sigma=0.1)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='wavelet', animation_filepath='./plots/contour_animation_wavelet_1.gif', sigma=1)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='wavelet', animation_filepath='./plots/contour_animation_wavelet_05.gif', sigma=0.5)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='wavelet', animation_filepath='./plots/contour_animation_wavelet_075.gif', sigma=0.75)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='wavelet', animation_filepath='./plots/contour_animation_wavelet_10.gif', sigma=10)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='laplace', animation_filepath='./plots/contour_animation_laplace.gif')
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='humphrey', animation_filepath='./plots/contour_animation_humphrey_05.gif', beta=0.5)
    get_contour_animation(scan_results, scan_parameters, t_start=t_start, t_end=t_end, t_stride=t_stride, denoise='humphrey', animation_filepath='./plots/contour_animation_humphrey_1.gif', beta=1.0)
