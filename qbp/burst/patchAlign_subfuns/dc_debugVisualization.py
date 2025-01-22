import numpy as np
import cv2
import os
from qbp.utils.flow_eval.drawFlowHSV import draw_flow_hsv
from testing.TestFunctions import test_logger


@test_logger
def debug_visualization(flow, imv, H, W, param, resultDir, i, timeBlockStart, patchSize, numStrides):
	X, Y = np.meshgrid(np.arange(W), np.arange(H))
	flowwarp = np.repeat(np.repeat(flow[0::numStrides, 0::numStrides, :], patchSize, axis=0), patchSize, axis=1)
	flowhsv = draw_flow_hsv(flowwarp)
	cv2.imwrite(os.path.join(resultDir, f'flow{i}-l1.png'), flowhsv)
	if imv.ndim == 2:
		imvWarped = cv2.remap(imv, (X + flowwarp[:, :, 0]).astype(np.float32),
							  (Y + flowwarp[:, :, 1]).astype(np.float32), cv2.INTER_CUBIC)
	else:
		imvWarped = np.zeros_like(imv)
		for c in range(imv.shape[2]):
			imvWarped[:, :, c] = cv2.remap(imv[:, :, c], (X + flowwarp[:, :, 0]).astype(np.float32),
										   (Y + flowwarp[:, :, 1]).astype(np.float32), cv2.INTER_CUBIC)
	cv2.imwrite(os.path.join(resultDir, f'imWarped{i}-l1.png'), imvWarped)
	elapsed = (cv2.getTickCount() - timeBlockStart) / cv2.getTickFrequency()
	print(f'{elapsed}')
