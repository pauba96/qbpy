import numpy as np
from qbpy.single_photon_imaging.src.window_fns.box_window_2D import box_window_2D

def idft_2D(Y, window_fn=None):
    """
    Perform a 2D Inverse Discrete Fourier Transform (IDFT) on an input array `Y`,
    optionally applying a windowing function correction.
    Note: the input data for this function cannot be saved using pickle, since it contains a function handle

    Parameters:
    - Y: 2D NumPy array (Fourier-transformed array)
    - window_fn: A callable window function that takes two arguments (H, W)
                 and returns a 2D window array of the same shape as `Y`.
                 If None, defaults to a box window (no windowing effect).
    - eng: Optional MATLAB engine to verify the IDFT against MATLAB's `ifft2`.

    Returns:
    - y: Real part of the inverse 2D Fourier-transformed array, optionally corrected by the window.
    """
    # Perform the 2D IFFT and take the real part
    y = np.real(np.fft.ifft2(Y))
    if window_fn is not None and window_fn != box_window_2D:
        # Apply window correction if window_fn is not the default box window
        H, W = Y.shape
        window = window_fn(H, W)
        y = y / window
    return y
