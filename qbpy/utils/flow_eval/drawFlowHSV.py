import numpy as np
import cv2


def draw_flow_hsv(flow, max_mag=None):
	"""
	Draw flow HSV plot as images.

	Args:
		flow (numpy array): Optical flow array with shape (H, W, 2).
		max_mag (float, optional): Maximum magnitude for normalization. If None, it is computed from the flow.

	Returns:
		numpy array: RGB image representing the flow in HSV color space.
	"""
	flowx = flow[:, :, 0]
	flowy = flow[:, :, 1]
	angle = np.arctan2(flowy, flowx)
	angle[angle < 0] += 2 * np.pi
	h = angle / (2 * np.pi)

	mag = np.sqrt(flowx ** 2 + flowy ** 2)
	if max_mag is None:
		max_mag = np.max(mag)
	s = mag / max_mag

	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
	hsv[:, :, 0] = h
	hsv[:, :, 1] = s
	hsv[:, :, 2] = 1

	flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return flow_rgb