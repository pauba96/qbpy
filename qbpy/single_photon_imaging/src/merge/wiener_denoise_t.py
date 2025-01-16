import numpy as np
import unittest
import os
import pickle
from qbpy.burst.patchAlign_subfuns.dc_utils import save_to_mat
from qbpy.single_photon_imaging.src.window_fns.raised_cos_window_2D import raised_cos_window_2D
from qbpy.single_photon_imaging.src.window_fns.box_window_2D import box_window_2D
from testing.io import get_eng
from qbpy.single_photon_imaging.src.sigproc.dft_2D import dft_2D
from qbpy.single_photon_imaging.src.sigproc.idft_2D import idft_2D
from testing.TestFunctions import test_logger


@test_logger
def wiener_denoise_t(patch_stack, c0, window_fn=None):
    """Denoises an aligned patch stack using Wiener filtering along the time axis.

    Args:
        patch_stack (numpy.ndarray): 3D array (H, W, D) of patches.
        c0 (float): Wiener filtering tuning parameter.
        window_fn (callable, optional): A function to generate a window. Defaults to a box window.

    Returns:
        numpy.ndarray: The denoised 2D array.
    """

    if window_fn is None:
        window_fn = box_window_2D  # Default to a box window.

    D = patch_stack.shape[2]
    base_frame = patch_stack[:, :, 0]
    noise_variance = max(np.finfo(float).eps, np.std(base_frame, ddof=1) ** 2)

    FT_base = dft_2D(base_frame, window_fn)
    FT_merged = FT_base.copy()
    c = patch_stack.shape[0] * patch_stack.shape[1] * 2 * c0

    for i in range(1, D):
        FT_frame = dft_2D(patch_stack[:, :, i], window_fn)
        Dz2 = np.abs(FT_frame - FT_base) ** 2
        Az = Dz2 / (Dz2 + c * noise_variance)
        FT_merged += Az * FT_base + (1 - Az) * FT_frame

    FT_merged /= D
    merged = idft_2D(FT_merged, window_fn)
    return merged