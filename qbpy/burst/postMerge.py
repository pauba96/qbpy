import numpy as np
from qbpy.utils.mleImage import mle_image
from testing.TestFunctions import test_logger

@test_logger
def post_merge(S, param, isSR=False, dcr=None):
	if isSR:
		raise NotImplementedError("Super resolution is not implemented in python")
	else:
		T = param["mergeTWNum"] * param["mergeTWSize"] * param["n_binary"]

	if dcr is not None:
		raise NotImplementedError("DCR is not implemented in python")
	S[S<0] = 0

	if param["bm3dSigma"] >0:
		raise NotImplementedError("BM3D is not implemented in python")
	else:
		imd = S;

	# invert the response curve
	imr, _ = mle_image(imd, T, param["imgScale"], True)  # Call the function
	# Restrict to the real part (if complex)
	imr = np.real(imr)
	# Clamp negative values to 0
	imr[imr < 0] = 0
	return imr