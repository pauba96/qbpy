import numpy as np
from qbpy.burst.blockMatch2d import block_match_2d_multichannel
from qbpy.utils.lkAlign import lk_align
from testing.TestFunctions import test_logger


@test_logger
def find_best_matches(refImg, tgtImg, patchSize, searchRadius, initMatch, param):
    dataType = param['dataType']
    hl, wl = initMatch.shape[:2]
    bestMatch = np.zeros((hl, wl, 2), dtype=dataType)

    # loop over templates -> refImg patch is only defined by lb and ub
    for j in range(hl):
        for k in range(wl):
            ylb = j * patchSize
            xlb = k * patchSize
            bestScore = float('inf')
            currBest = np.zeros(2)

            for m in range(initMatch.shape[2]):
                currMatch, currScore = block_match_2d_multichannel(refImg, tgtImg, [ylb+1, xlb+1],  # +1 because function expects Matlab indexing
                                                                   patchSize, searchRadius,
                                                                   initMatch[j, k, m, :])
                if currScore < bestScore:
                    currBest = currMatch
                    bestScore = currScore

            # LK refinement
            tempf = lk_align(refImg[ylb:ylb + patchSize, xlb:xlb + patchSize],
                             tgtImg,
                             param['numLKIters'],
                             currBest + np.array([xlb, ylb]).astype(np.float32))

            bestMatch[j, k, :] = tempf - np.array([xlb, ylb])

    return bestMatch