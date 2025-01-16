import numpy as np
import os
import unittest
import pickle
from qbpy.burst.buildAgrePyramid import build_aggre_pyramid
from qbpy.utils.ps_shape.param_from_json import param_from_json
from testing.io import get_eng
from testing.TestFunctions import test_logger


@test_logger
def initialize_matches_from_level2(bestMatch, patchSizes, patchStride, param, ref_size):
    dataType = param['dataType']
    # Compute dimensions based on bestMatch shape and patch sizes
    H = ref_size[0]
    W = ref_size[1]
    hs = (H - patchSizes[0]) // patchStride + 1
    ws = (W - patchSizes[0]) // patchStride + 1
    initMatch = np.zeros((hs, ws, 3, 2), dtype=dataType)

    if param['numLevels'] > 1:
        bestMatch = np.round(bestMatch * param['upsampleRatios'][1]) #this is to prepare for the final level, hence the magic number 1
        yv = np.arange(0, H - patchSizes[0] + 1, patchStride) / param['upsampleRatios'][1] / patchSizes[1]
        xv = np.arange(0, W - patchSizes[0] + 1, patchStride) / param['upsampleRatios'][1] / patchSizes[1]

        yr = np.floor(yv).astype(int)
        xr = np.floor(xv).astype(int)

        grid_y, grid_x = np.meshgrid(yr, xr, indexing='ij')
        initMatch[:, :, 0, 0] = bestMatch[grid_y, grid_x, 0]
        initMatch[:, :, 0, 1] = bestMatch[grid_y, grid_x, 1]

        yrn = yr - 1
        masky = ((yv % 1 >= 0.5) & (yr < bestMatch.shape[0]-1)) | (yr == 0)
        yrn[masky] = yr[masky] + 1
        grid_y, grid_x = np.meshgrid(yrn, xr, indexing='ij')
        initMatch[:, :, 1, 1] = bestMatch[grid_y, grid_x, 1]
        initMatch[:, :, 1, 0] = bestMatch[grid_y, grid_x, 0]

        xrn = xr - 1
        maskx = ((xv % 1 >= 0.5) & (xr < bestMatch.shape[1] - 1)) | (xr == 0)
        xrn[maskx] = xr[maskx] + 1  # shift right by one
        grid_y, grid_x = np.meshgrid(yr, xrn, indexing='ij')
        initMatch[:, :, 2, 1] = bestMatch[grid_y, grid_x, 1]
        initMatch[:, :, 2, 0] = bestMatch[grid_y, grid_x, 0]

    return initMatch
