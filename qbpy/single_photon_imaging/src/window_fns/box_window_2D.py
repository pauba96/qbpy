"""
Matlab implementation:

function [w] = box_window_2D(M, N)
%BOX_WINDOW_2D
    w = ones(M, N);
end
"""
import numpy as np
from testing.TestFunctions import test_logger

@test_logger
def box_window_2D(M, N):
    """Generates a box window in 2D.

    Args:
        M (int): Number of rows.
        N (int): Number of columns.

    Returns:
        numpy.ndarray: A box window of size (M, N).
    """
    return np.ones((M, N))