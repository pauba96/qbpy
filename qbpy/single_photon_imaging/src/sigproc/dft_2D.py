import numpy as np
from qbpy.single_photon_imaging.src.window_fns.box_window_2D import box_window_2D

def dft_2D(y, window_fn=None):
    """
    Perform a 2D Discrete Fourier Transform (DFT) on an input array `y`,
    optionally applying a windowing function.
    Note: the input data for this function cannot be saved using pickle, since it contains a function handle

    Parameters:
    - y: 2D NumPy array
    - window_fn: A callable window function that takes two arguments (H, W)
                 and returns a 2D window array of the same shape as `y`.
                 If None, defaults to a box window (no windowing effect).

    Returns:
    - Y: 2D Fourier-transformed array
    """
    y = np.ascontiguousarray(y)
    if window_fn is None:
        # Default to box window (no effect)
        window_fn = box_window_2D
    # Ensure the input is a floating-point array
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float64)
    # Apply the window function if it's not the default box window
    if window_fn != box_window_2D:
        H, W = y.shape
        window = window_fn(H, W)
        m = np.mean(y)
        y = m + (y - m) * window
    # Perform the 2D FFT
    Y = np.fft.fft2(y)
    return Y