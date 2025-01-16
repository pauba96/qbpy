import numpy as np
from testing.TestFunctions import test_logger


@test_logger
def mle_image(S, T, scale=1, fix_inf=False):
	"""
	Convert a sum image into a linear intensity image.

	Args:
		S (numpy array): Sum image. Always 3 channels (3rd may be size 1)
		T (int or numpy array): Total number of frames.
		scale (float or numpy array, optional): Linear scaling factor for intensity image. Default is 1.
		fix_inf (bool, optional): Whether to fix infinite values. Default is False.

	Returns:
		tuple: (Lambda, sigma2)
			- Lambda (numpy array): Linear intensity image.
			- sigma2 (numpy array): Variance image (if applicable).
	"""
	if len(S.shape) == 2:
		S = S[:, :, np.newaxis]
	if fix_inf:
		if np.isscalar(T):
			S[S > T - 1] = T - 1
		else:
			S[S > T - 1] = T[S > T - 1] - 1

	# check, if np.log(1 - S / T) is divide by zero
	if np.any(S == T):
		print("Warning: S == T, division by zero in mle_image")
		#S[S == T] += 0.1
	Lambda = -np.log(1 - S / T)

	if scale is not None:
		if Lambda.shape[2] == 1:
			scale = np.mean(scale)
			Lambda *= scale
		else:
			if np.size(scale) == 3:
				Lambda *= np.reshape(scale, (1, 1, np.size(scale)))
			else:
				Lambda *= scale
	else:
		Lambda /= np.max(Lambda)

	sigma2 = None
	if fix_inf:
		LambdaFixed = np.copy(Lambda)
		LambdaFixed[LambdaFixed == 0] = -np.log(1 - 1 / T)
		sigma2 = (np.exp(LambdaFixed) - 1) / (T)

	return Lambda, sigma2