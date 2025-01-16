import numpy as np
import unittest
import os
import pickle
from qbpy.burst.buildAgrePyramid import build_aggre_pyramid
from qbpy.utils.ps_shape.param_from_json import param_from_json
from qbpy.burst.blockMatch2d import block_match_2d_multichannel
from qbpy.utils.lkAlign import lk_align
from qbpy.burst.patchAlign_subfuns.initializeMatchesFromLevel2 import initialize_matches_from_level2
from qbpy.burst.patchAlign_subfuns.dc_utils import save_to_mat
from testing.io import get_eng
from testing.TestFunctions import test_logger


@test_logger
def refine_finest_level(refImg, tgtImg, bestMatch, patchSizes, patchStride, searchRadius, param):
    dataType = param['dataType']
    H, W = refImg.shape[:2]
    hs = (H - patchSizes[0]) // patchStride + 1
    ws = (W - patchSizes[0]) // patchStride + 1


    initMatch = initialize_matches_from_level2(bestMatch, patchSizes, patchStride, param, refImg.shape)

    finalFlow = np.zeros((hs, ws, 2), dataType)

    for j in range(hs):
        for k in range(ws):
            ylb = j * patchStride
            xlb = k * patchStride
            bestScore = float('inf')
            currBest = np.zeros(2)
            for m in range(initMatch.shape[2]):
                currMatch, currScore = block_match_2d_multichannel(refImg, tgtImg, [ylb +1, xlb+1], # +1 because function expects Matlab indexing
                                                                   patchSizes[0], searchRadius,
                                                                   initMatch[j, k, m, :])
                if currScore < bestScore:
                    currBest = currMatch
                    bestScore = currScore

            if not (param['fastMode'] and not param['doSR']):
                tempf = lk_align(refImg[ylb:ylb + patchSizes[0], xlb:xlb + patchSizes[0]],
                                 tgtImg,
                                 param['numLKIters'],
                                 currBest + np.array([xlb, ylb]))
                finalFlow[j, k, :] = tempf - np.array([xlb, ylb])
            else:
                finalFlow[j, k, :] = currBest

    return finalFlow