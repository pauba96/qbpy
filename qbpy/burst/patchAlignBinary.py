import numpy as np
import os
from qbpy.utils.mleImage import mle_image
import h5py
from qbpy.burst.patchAlign import patch_align
from qbpy.burst.patchAlignRefine import patch_align_refine
from testing.TestFunctions import test_logger


@test_logger
def patch_align_binary(imbs, param):
    """
    Align binary sequence using patchAlign.

    Args:
        imbs (list): 1D list of binary frames (numpy arrays).
        param (dict): Dictionary containing the following fields:
            - alignTWSize: window size for temporal reconstruction.
            - alignTWNum: number of temporal windows (determines total number of frames being used), > 2.
            - numLevels: number of pyramid levels.
            - patchSizes: array that contains patch sizes for each level.
            - upsampleRatios: array that contains the upsample ratios for each pyramid level (first entry always 1).
            - searchRadii: array that contains the search radius at each level.
            - numLKIters: number of Lucas-Kanade iterations for subpixel refinement.
            - refFrame: reference frame #.
            - imgScale: linear scaling factor for intensity image.
            - doRefine: do flow refinement.
            - resultDir: directory to save results in.
            - debug: whether or not to print debug information.

    Returns:
        tuple: (flows, flowrs)
            - flows: 1D list of computed flows.
            - flowrs: 1D list of refined flows.
    """
    N = len(imbs)
    H = imbs[0].shape[0]
    W = imbs[0].shape[1]
    C = imbs[0].shape[2] if len(imbs[0].shape) == 3 else 1
    alignTWSize = param['alignTWSize']
    alignTWNum = param['alignTWNum']
    if alignTWSize * alignTWNum > N:
        raise ValueError('alignTWSize * alignTWNum must be no greater than N!')

    refFrame = param['refFrame'] # in matlab indexing
    if alignTWSize * alignTWNum < refFrame:
        raise ValueError('alignTWSize * alignTWNum must be no smaller than refFrame!')
    refBlock = (refFrame - 1) // alignTWSize + 1  # in matlab indexing!
    param['refImage'] = refBlock

    def frame_idx(i, j):
        return (i - 1) * alignTWSize + j

    imgScale = param['imgScale']
    resultDir = param['resultDir']

    blockAggres = []
    for i in range(1, alignTWNum + 1):
        S = np.zeros((H, W), dtype=param['dataType'])
        # get blocks by merging TWSize number of individual frames
        for j in range(1, alignTWSize + 1):
            if C == 1:
                S += imbs[frame_idx(i, j) - 1]
            else:
                S += np.mean(imbs[frame_idx(i, j) - 1], axis=2)
        blockAggres.append(S / alignTWSize)

    if param['debug']:
        blockRecons = []
        for i in range(1, alignTWNum + 1):
            blockRecons.append(mle_image(blockAggres[i - 1] * alignTWSize, alignTWSize, imgScale, True)[0])

            os.makedirs(resultDir, exist_ok=True)
    else:
        blockRecons = []

    flows = patch_align(blockAggres, param, blockRecons)


    if param['doRefine']:
        raise NotImplementedError("doRefine not implemented yet")
        flowrs = patch_align_refine(blockAggres, flows, param, blockRecons)
    else:
        flowrs = []

    print('Alignment done.')
    return flows, flowrs

