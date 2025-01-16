import numpy as np
import cv2
from testing.TestFunctions import test_logger
@test_logger
def build_aggre_pyramid(ims, upsample_ratios):
    """
    Build aggregate image pyramid. Coarser levels are built by aggregating photons from finer levels.
    Assume images are already normalized to 0...1.

    Args:
        ims (numpy array): Input image.
        upsample_ratios (list): List of upsample ratios for each pyramid level.

    Returns:
        list: List of images representing the pyramid.
    """
    num_levels = len(upsample_ratios)
    P = [None] * num_levels
    P[0] = ims
    S0 = ims
    aggre_ratio = 1

    for i in range(1, num_levels):
        scale_filter = np.ones((upsample_ratios[i], upsample_ratios[i]), dtype=np.float32)
        scale_center = (upsample_ratios[i] + 1) // 2
        S0 = cv2.filter2D(S0, -1, scale_filter)
        S0 = S0[scale_center::upsample_ratios[i], scale_center::upsample_ratios[i]]
        aggre_ratio *= upsample_ratios[i] ** 2
        P[i] = S0 / aggre_ratio

    return P