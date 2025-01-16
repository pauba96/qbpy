import numpy as np
import cv2
import os
from qbpy.utils.flow_eval.drawFlowHSV import draw_flow_hsv
from testing.TestFunctions import test_logger


@test_logger
def visualize_flow(bestMatch, size_next_level, resultDir, level):
	# Visualize intermediate flow for debugging
	H, W = size_next_level[:2]
	flowwarp = np.repeat(bestMatch, H // bestMatch.shape[0], axis=0)
	flowwarp = np.repeat(flowwarp, W // bestMatch.shape[1], axis=1)
	flowhsv = draw_flow_hsv(flowwarp)
	cv2.imwrite(os.path.join(resultDir, f'flow-temp-l{level}.png'), flowhsv)