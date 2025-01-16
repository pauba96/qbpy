import numpy as np
import os
import unittest
import pickle
from qbpy.burst.patchAlign_subfuns.dc_findBestMatches import find_best_matches
from qbpy.utils.ps_shape.param_from_json import param_from_json
from qbpy.burst.buildAgrePyramid import build_aggre_pyramid
from qbpy.burst.patchAlign_subfuns.dc_upsampleMatches import upsample_matches
from qbpy.burst.patchAlign_subfuns.dc_utils import save_to_mat
from testing.io import get_eng
from testing.TestFunctions import test_logger


@test_logger
def coarse_to_fine_match(P0, P1, patchSizes, searchRadii, upsampleRatios, param):
    numLevels = param['numLevels']
    dataType = param['dataType']
    l = numLevels - 1
    hl = P1[l].shape[0] // patchSizes[l]
    wl = P1[l].shape[1] // patchSizes[l]
    initMatch = np.zeros((hl, wl, 1, 2), dtype=dataType)
    bestMatch = None

    for level in range(numLevels - 1, 0, -1):
        if param['debug']:
            print(f'L{level}:', end='')

        ### Match patches at current level
        bestMatch = find_best_matches(P0[level], P1[level], patchSizes[level], searchRadii[level],
                                      initMatch, param)
        bestMatch_in = bestMatch.copy()

        ### Upsample matches from previous level
        if level > 1:
            initMatch, bestMatch = upsample_matches(bestMatch_in, P1[level - 1],
                                                    patchSizes[level - 1], patchSizes[level],
                                                    upsampleRatios[level], dataType)


    return bestMatch
