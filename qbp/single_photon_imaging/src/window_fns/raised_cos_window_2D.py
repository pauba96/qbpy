import numpy as np
import os
import pickle
import unittest
from qbp.burst.patchAlign_subfuns.dc_utils import save_to_mat
from testing.io import get_eng
from testing.TestFunctions import test_logger


def raised_cos_window_1D(L):
    """Generates a raised cosine window in 1D.
    Args:
        L (int): Length of the window.
    Returns:
        numpy.ndarray: The raised cosine window of length L.
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(L) + 0.5) / L)

@test_logger
def raised_cos_window_2D(M,N):
    """Generates a raised cosine window in 2D.

    Args:
        M (int): Number of rows.
        N (int): Number of columns.

    Returns:
        numpy.ndarray: The raised cosine window of size (M, N).
    """
    wv = raised_cos_window_1D(M)
    wh = raised_cos_window_1D(N)
    return np.outer(wv, wh)
