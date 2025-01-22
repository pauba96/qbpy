import numpy as np
import unittest
import os
import pickle
from qbp.burst.buildAgrePyramid import build_aggre_pyramid
from qbp.utils.ps_shape.param_from_json import param_from_json
from testing.io import get_eng
from testing.TestFunctions import test_logger


@test_logger
def upsample_matches(bestMatch, tgtImg, finerPatchSize, coarserPatchSize, ratio, dataType):
    """
    Args:
        bestMatch: Contains the results of the previous matching step
        tgtImg:
        finerPatchSize:
        coarserPatchSize:
        ratio:
        dataType:

    Returns:

    Notes: bestMatch
    """

    hl = tgtImg.shape[0] // finerPatchSize
    wl = tgtImg.shape[1] // finerPatchSize
    initMatch = np.zeros((hl, wl, 3, 2), dtype=dataType)

    bestMatch = np.round(bestMatch * ratio)
    yv = np.arange(hl) * finerPatchSize / ratio / coarserPatchSize
    xv = np.arange(wl) * finerPatchSize / ratio / coarserPatchSize
    yr = np.floor(yv).astype(int)
    xr = np.floor(xv).astype(int)

    # Create a grid of indices to ensure all combinations are covered
    grid_y, grid_x = np.meshgrid(yr, xr, indexing='ij')
    initMatch[:, :, 0, 1] = bestMatch[grid_y, grid_x, 1]
    initMatch[:, :, 0, 0] = bestMatch[grid_y, grid_x, 0]

    yrn = yr - 1
    # masking is performed to account for fractional positions. If true, a pixel is shifted up, if false, it is not.
    masky = ((yv % 1 >= 0.5) & (yr < bestMatch.shape[0]-1)) | (yr == 0)
    yrn[masky] = yr[masky] + 1 # shift up by one
    grid_y, grid_x = np.meshgrid(yrn, xr, indexing='ij')
    initMatch[:, :, 1, 1] = bestMatch[grid_y, grid_x, 1]
    initMatch[:, :, 1, 0] = bestMatch[grid_y, grid_x, 0]

    xrn = xr - 1
    maskx = ((xv % 1 >= 0.5) & (xr < bestMatch.shape[1]-1)) | (xr == 0)
    xrn[maskx] = xr[maskx] + 1 # shift right by one
    grid_y, grid_x = np.meshgrid(yr, xrn, indexing='ij')
    initMatch[:, :, 2, 1] = bestMatch[grid_y, grid_x, 1]
    initMatch[:, :, 2, 0] = bestMatch[grid_y, grid_x, 0]


    return initMatch, bestMatch