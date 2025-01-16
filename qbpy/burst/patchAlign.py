import numpy as np
import cv2
from qbpy.burst.buildAgrePyramid import build_aggre_pyramid
from qbpy.burst.patchAlign_subfuns.dc_coarseToFineMatch import coarse_to_fine_match
from qbpy.burst.patchAlign_subfuns.dc_refineFinestLevel import refine_finest_level
from qbpy.burst.patchAlign_subfuns.dc_debugVisualization import debug_visualization
from testing.TestFunctions import test_logger

@test_logger
def patch_align(ims, param, imv=None):
    """
    Python equivalent of patchAlign. Uses helper functions.
    """
    H, W = ims[0].shape[:2]
    C = ims[0].shape[2] if ims[0].ndim == 3 else 1
    N = len(ims)
    refImage = param['refImage'] - 1 #convert indexing to python

    if imv is None:
        imv = ims

    # Convert to grayscale if needed
    if C == 1:
        img = ims
    else:
        img = [np.mean(im,axis=(0,1)) for im in ims]

    resultDir = param['resultDir']
    numLevels = param['numLevels']
    patchSizes = param['patchSizes']
    patchStride = patchSizes[0] // 2
    upsampleRatios = param['upsampleRatios']
    searchRadii = param['searchRadii']
    numStrides = patchSizes[0] // patchStride
    hs = (H - patchSizes[0]) // patchStride + 1
    ws = (W - patchSizes[0]) // patchStride + 1
    flows = [None] * N

    P0 = build_aggre_pyramid(img[refImage], upsampleRatios)

    for i in range(N):
        if i == refImage:
            flows[i] = np.zeros((hs, ws, 2))
            continue

        if param['debug']:
            print(f'Block {i}: ', end='')
        timeBlockStart = cv2.getTickCount()

        P1 = build_aggre_pyramid(img[i], upsampleRatios)

        ### Coarse-to-fine matching
        bestMatch = coarse_to_fine_match(P0, P1, patchSizes, searchRadii, upsampleRatios, param)

        ### Refine at finest level
        flows[i] = refine_finest_level(P0[0], P1[0], bestMatch, patchSizes, patchStride, searchRadii[0], param)

        # Debug visualization
        if param['debug']:
            debug_visualization(flows[i], imv[i], H, W, param, resultDir, i, timeBlockStart, patchSizes[0], numStrides)

    return flows