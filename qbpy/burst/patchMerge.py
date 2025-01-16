import numpy as np
from qbpy.single_photon_imaging.src.window_fns.raised_cos_window_2D import raised_cos_window_2D
from qbpy.single_photon_imaging.src.merge.wiener_denoise_t import wiener_denoise_t
from testing.TestFunctions import test_logger


@test_logger
def patch_merge(patches, param):
	"""
    Merge pre-aligned intensity patches via patchwise Wiener filtering.

    Args:
        patches: 4D numpy array of pre-aligned blocks of size
                 (hs * patchSize, ws * patchSize, C, N)
        param: dict containing:
            - refImage: index of the reference image in ims
            - patchSizes: array of patch sizes for each level
            - wienerC: tuning parameter C for Wiener filtering
            - debug: whether or not to print debug information
            - H: height of the final output image
            - W: width of the final output image
            - dataType: numpy dtype for the final output

    Returns:
        S: Merged image of size (H, W, C)
    """
	patches = patches.copy()
	param = param.copy()

	Hp, Wp, C, N = patches.shape
	H = int(param['H'])
	W = int(param['W'])
	patchSize = int(param['patchSizes'][0])
	patchStride = patchSize // 2

	refImage = param['refImage']

	winWeights = raised_cos_window_2D(patchSize, patchSize)

	imo = np.zeros((H, W, C), dtype=param['dataType'])
	hs = Hp // patchSize
	ws = Wp // patchSize

	for c in range(C):
		# Swap the central block and the first block
		patches_ = patches.copy()
		patches[:, :, c, 0] = patches_[:, :, c, refImage-1]
		patches[:, :, c, refImage-1] = patches_[:, :, c, 0]

		# Merge all blocks
		Sa = np.zeros((H, W), dtype=param['dataType'])
		accWeights = np.zeros((H, W), dtype=param['dataType'])
		if param['debug']:
			print('Block merging...')

		for i in range(hs):
			if param['debug']:
				print(f"{i}", end="")
			for j in range(ws):
				ylb = i * patchSize
				xlb = j * patchSize
				patch_stack = patches[ylb:ylb + patchSize, xlb:xlb + patchSize, c, :]
				Sdenoised = wiener_denoise_t(patch_stack, param['wienerC'])

				ylb = i * patchStride
				xlb = j * patchStride

				Sa[ylb:ylb + patchSize, xlb:xlb + patchSize] += Sdenoised * winWeights
				accWeights[ylb:ylb + patchSize, xlb:xlb + patchSize] += winWeights

				if param['debug']:
					print('.', end="")
			if param['debug']:
				print()

		Sa = Sa / accWeights
		imo[:, :, c] = Sa

	imo[imo < 0] = 0
	return imo