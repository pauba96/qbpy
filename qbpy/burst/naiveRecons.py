import numpy as np
from qbpy.utils.mleImage import mle_image
from testing.TestFunctions import test_logger
@test_logger
def naive_recons(imbs, param):
    """
    Naive reconstruction by simple summing the binary images and compute MLE.
    V2: use mle_image instead of mle_intensity, ignores absolute intensity.

    Args:
        imbs (list): 1D list of binary frames (numpy arrays).
        param (dict): Dictionary containing the following fields:
            - mergeTWSize: window size for temporal reconstruction.
            - mergeTWNum: number of temporal windows (determines total number of frames being used), > 2.
            - refFrame: reference frame #.
            - imgScale: linear scaling factor for intensity image.
            - debug: whether or not to print debug information.

    Returns:
        tuple: (ima, S)
            - ima: reconstructed image.
            - S: sum image.
    """
    N = len(imbs)
    # Get the dimensions of the first image, if its two dimensional, set C to 1
    H = imbs[0].shape[0]
    W = imbs[0].shape[1]
    C = imbs[0].shape[2] if len(imbs[0].shape) == 3 else 1

    tw_size = param['mergeTWSize']
    tw_num = param['mergeTWNum']

    if tw_size * tw_num > N:
        raise ValueError('twSize * twNum must be no greater than N!')

    def frame_idx(i, j):
        return (i - 1) * tw_size + j

    img_scale = param['imgScale']

    S = np.zeros((H, W, C), dtype=param['dataType'])
    for i in range(1, tw_num + 1):
        for j in range(1, tw_size + 1):
            if C==1:
                S += imbs[frame_idx(i, j) - 1][:,:,np.newaxis]
            else:
                S += imbs[frame_idx(i, j) - 1]

    ima, sigma2 = mle_image(S, tw_num * tw_size * param['n_binary'], img_scale)

    return ima.squeeze(), S
