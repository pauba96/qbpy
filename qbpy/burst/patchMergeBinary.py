import numpy as np
from qbpy.utils.interp2 import interp2
from qbpy.burst.patchMerge import patch_merge
from testing.TestFunctions import test_logger


@test_logger
def patch_merge_binary(imbs, flows, param, phase_ids):
    # clean inputs to accept both matlab and python inputs
    if not isinstance(imbs[0], np.ndarray):
        imbs = [np.array(im) for im in imbs]
        flows = [np.array(flow) for flow in flows]
        phase_ids = np.array(phase_ids)

    # Parameters
    H, W = imbs[0].shape
    C = 1 if len(imbs[0].shape) == 2 else imbs.shape[2]
    patchSize = int(param['patchSizes'][0])
    patchStride = int(patchSize / 2)
    alignTWSize = int(param['alignTWSize'])
    alignTWNum = int(param['alignTWNum'])
    mergeTWSize = int(param['mergeTWSize'])
    mergeTWNum = int(param['mergeTWNum'])
    num_ls = int(param['num_ls'])

    if mergeTWSize * mergeTWNum > alignTWSize * alignTWNum:
        raise ValueError('mergeTWSize * mergeTWNum must be <= alignTWSize * alignTWNum')

    refFrame = param['refFrame']
    if mergeTWSize * mergeTWNum < refFrame:
        raise ValueError('mergeTWSize * mergeTWNum must be >= refFrame')

    refBlock = int(np.floor((refFrame - 1) / mergeTWSize)) + 1
    param['refImage'] = refBlock

    def frameIdx(i, j):
        return (i - 1) * mergeTWSize + j

    def alignSub(idx):
        i = np.floor((idx - 1) / alignTWSize) + 1
        j = np.mod(idx - 1, alignTWSize) + 1
        return i, j

    imgScale = param['imgScale']

    if alignTWNum == 1:
        assert mergeTWNum == 1 and alignTWSize == mergeTWSize
        S = np.zeros(imbs[0].shape, dtype=param['dataType'])
        for i in range(0,alignTWSize):
            S += imbs[frameIdx(1, i)]
        print("Merging done.")
        return S

    alignCenFrame = (refFrame - 1) % alignTWSize + 1

    flowsr = [None] * int(alignTWNum + 2)
    for i in range(int(alignTWNum)):
        flowsr[i + 1] = flows[i]
    flowsr[0] = 2 * flowsr[1] - flowsr[2]
    flowsr[-1] = 2 * flowsr[-2] - flowsr[-3]

    def interpFlow(i, j):
        idx = frameIdx(i, j)
        ai, aj = alignSub(idx)
        if aj < alignCenFrame:
            return ((alignCenFrame - aj) / alignTWSize) * flowsr[int(ai) - 1] + \
                   ((aj + alignTWSize - alignCenFrame) / alignTWSize) * flowsr[int(ai)]
        else:
            return ((alignCenFrame + alignTWSize - aj) / alignTWSize) * flowsr[int(ai)] + \
                   ((aj - alignCenFrame) / alignTWSize) * flowsr[int(ai) + 1]

    hs = int((H - patchSize) / patchStride + 1)
    ws = int((W - patchSize) / patchStride + 1)
    yv = np.repeat(np.arange(0, hs) * patchStride, patchSize) + np.tile(np.arange(1, patchSize + 1), hs)
    xv = np.repeat(np.arange(0, ws) * patchStride, patchSize) + np.tile(np.arange(1, patchSize + 1), ws)
    X, Y = np.meshgrid(xv, yv)

    if phase_ids is not None and np.any(phase_ids):
        blockPatches = np.zeros((int(hs * patchSize), int(ws * patchSize), num_ls, mergeTWNum), dtype=param['dataType'])
        # Readjust images using the per-frame flow
        for i in range(1, mergeTWNum + 1):
            Sb = np.zeros_like(blockPatches[:, :, :, 0])
            countMap = np.zeros_like(Sb)

            for j in range(1, mergeTWSize + 1):
                phase = int(phase_ids[frameIdx(i, j) - 1])
                curFlow = interpFlow(i, j)
                flowwarp = np.repeat(np.repeat(curFlow, patchSize, axis=0), patchSize, axis=1)

                curFrame = np.array(imbs[frameIdx(i, j) - 1], dtype=float)

                if param["fastMode"]:
                    method = 'nearest'
                else:
                    method = 'linear'
                imbwarped = interp2(curFrame, X + flowwarp[:, :, 0], Y + flowwarp[:, :, 1], method=method, indexing="mat")

                countMap[:, :, phase] += np.isfinite(imbwarped)
                imbwarped = np.array(imbwarped)
                imbwarped[np.isnan(imbwarped)] = 0
                Sb[:, :, phase] += imbwarped

            countMap[countMap == 0] = 1 # avoid division by zero, if countMap is 0, the corresponding pixel will also be 0
            Sb /= countMap
            blockPatches[:, :, :, i - 1] = Sb
    else: #  standard matching without phases
        blockPatches = np.zeros((hs * patchSize, ws * patchSize, C, mergeTWNum), dtype=param['dataType'])
        for c in range(C):
            print(f'Pre-warping Channel {c + 1}...')
            for i in range(1,mergeTWNum+1):
                Sb = np.zeros((hs * patchSize, ws * patchSize), dtype=param['dataType'])
                countMap = np.zeros_like(Sb, dtype=param['dataType'])
                if param.get('debug', False):
                    print('.')
                if 'warpTWSize' in param and param['warpTWSize'] > 1:
                    # this code section allows to treat sets of frames in merging by taking their average. This is
                    # computationally more efficient, but introduces blur in the final image.
                    warpTWSize = param['warpTWSize']
                    mergeTWSize = param['mergeTWSize']
                    assert mergeTWSize % warpTWSize == 0, "mergeTWSize must be divisible by warpTWSize"
                    warpTWNum = mergeTWSize // warpTWSize
                    for j in range(warpTWNum):
                        curFlow = interpFlow(i, 1+((j) * warpTWSize + (warpTWSize + 1) // 2))
                        flowwarp = np.repeat(curFlow, patchSize, axis=0).repeat(patchSize, axis=1)
                        curFrame = np.zeros((H,W), dtype=param['dataType'])
                        for k in range(warpTWSize):  # sum of images along
                            frame_idx = frameIdx(i, j * warpTWSize + k)
                            if C == 1:
                                curFrame += imbs[frame_idx].astype(np.float64)
                            else:
                                curFrame += imbs[frame_idx][:, :, c].astype(np.float64)
                        interp = interp2(curFrame, X + flowwarp[:, :, 0], Y + flowwarp[:, :, 1], method='linear',
                                indexing="mat")
                        imbwarped = interp
                        countMap += np.isfinite(imbwarped) * warpTWSize
                        imbwarped[~np.isfinite(imbwarped)] = 0
                        Sb += imbwarped
                else:
                    mergeTWSize = param['mergeTWSize']
                    for j in range(1,mergeTWSize+1):
                        curFlow = interpFlow(i, j)
                        flowwarp = np.repeat(curFlow, patchSize, axis=0).repeat(patchSize, axis=1)
                        frame_idx = frameIdx(i, j)
                        if C==1:
                            curFrame = imbs[frame_idx-1].astype(np.float64)
                        else:
                            curFrame = imbs[frame_idx-1][:, :, c].astype(np.float64)
                        if param.get('fastMode', False):
                            interp = interp2(curFrame, X + flowwarp[:, :, 0], Y + flowwarp[:, :, 1], method='nearest',
                                    indexing="mat")
                        else:
                            interp = interp2(curFrame, X + flowwarp[:, :, 0], Y + flowwarp[:, :, 1], method='linear',
                                    indexing="mat")
                        imbwarped = interp
                        countMap += np.isfinite(imbwarped)
                        imbwarped[~np.isfinite(imbwarped)] = 0
                        Sb += imbwarped
                countMap[countMap == 0] = 1
                Sb /= countMap
                blockPatches[:, :, c, i-1] = Sb

    # Wiener merge
    param['H'] = H
    param['W'] = W
    S = patch_merge(blockPatches, param) # fine-alignment (LK) and merge
    S = np.array(S) * float(mergeTWNum * mergeTWSize)
    print("Merging done.")

    return S.squeeze()


