import os
import numpy as np
import pickle
from qbpy.utils.ps_shape.param_from_json import param_from_json
from qbpy.utils.ps_shape.load_dataset import load_dataset
from qbpy.burst.naiveRecons import naive_recons
from qbpy.burst.patchAlignBinary import patch_align_binary
from qbpy.burst.patchAlign_subfuns.dc_utils import save_to_mat
from qbpy.burst.patchMergeBinary import patch_merge_binary
from qbpy.burst.postMerge import post_merge


def run_qbp_stepwise_verbose(json_path, eng=None):
	os.environ["ENABLE_TEST_LOGGER"] = "1"  # Enable logging
	param = param_from_json(json_path)
	# Check if the parameters are equivalent
	# run python version of load_dataset and check if equal
	imbs_py, dcr_py, h5_info_py, dropped_py, phase_ids_py = load_dataset(param)

	if eng is not None:
		param_mat = eng.param_from_json(json_path)
		imbs_mat, dcr_mat, h5_info_mat, dropped_mat, phase_ids_mat = eng.load_dataset(param_mat, nargout=5)
		assert np.allclose(imbs_py, imbs_mat)
		# warning that dcr is not implemented, yet
		print("Warning: dcr is not implemented, yet")
		assert np.allclose(dropped_mat, dropped_py)
		if np.any(phase_ids_py):
			assert np.allclose(np.array(phase_ids_mat)[:,0], phase_ids_py)

	# Naive reconstruction
	ima_py, S = naive_recons(imbs_py, param)
	if eng is not None:
		ima_mat = eng.naiveRecons(imbs_mat, param_mat)
		assert np.allclose(ima_mat, ima_py)

	# reference block naive reconstruction
	asParam_py = param.copy()
	asParam_py["mergeTWNum"] = 1
	refBlock = np.floor((param["refFrame"] - 1) / param["mergeTWSize"])
	start_idx = int(refBlock * param["mergeTWSize"])
	stop_idx = int((refBlock + 1) * param["mergeTWSize"])
	sliced_imbs_py = imbs_py[start_idx:stop_idx]
	imas_py, S = naive_recons(sliced_imbs_py, asParam_py)
	if eng is not None:
		# naive reconstruction with mergeTWNum = 1 and only one temporal window
		asParam_mat = param_mat.copy()
		asParam_mat["mergeTWNum"] = 1.0
		sliced_imbs_mat = imbs_mat[start_idx:stop_idx]
		imas = eng.naiveRecons(sliced_imbs_mat, asParam_mat)
		assert np.allclose(imas, imas_py)

	# Remove hot pixels
	if param["removeHP"]:
		assert eng is not None
		for i in range(1, len(imbs_mat)):
			imbs_mat[i] = eng.removeHotPixels(imbs_mat[i], dcr_mat, param["hpThresh"])
		print("Finished hp correction")
		imaf = eng.naiveRecons(imbs_mat, param_mat)
		imasf = eng.naiveRecons(sliced_imbs_mat, asParam_mat)
		raise NotImplementedError("removeHP is not implemented in python")
	else:
		imaf = []
		imasf = []
	print('Finished naive reconstruction.')

	# Align
	try:
		#python version of patchAlignBinary
		flows, flowrs = patch_align_binary(imbs_py, param)
		if eng is not None:
			flows_mat, flowrs_mat = eng.patchAlignBinary(imbs_mat, param_mat, nargout=2)
			np.testing.assert_allclose(flows, flows_mat, equal_nan=True, atol=1e-10)
			assert np.allclose(flowrs, flowrs_mat)
	except Exception as e:
		print(f"Error in run_qbp: {e}")
		# save to mat
		if eng is not None:
			save_to_mat(eng,
						os.getenv("QBPY_BASE_DIR")+r"/testing/test_data_inputs/patch_align_binary_inputs.mat",
						imbs=imbs_mat, param=param_mat)
		with open(
				os.getenv("QBPY_BASE_DIR")+r"/testing/test_data_inputs/patch_align_binary_inputs.pkl",
				'wb') as f:
			pickle.dump(
				{'imbs': imbs_py, 'param': param, 'param_mat': param_mat}, f)
		# throw exception
		raise e

	# Merge
	Sr = patch_merge_binary(imbs_py, flows, param, phase_ids_py)
	if eng is not None:
		Sr_mat = eng.patchMergeBinary(imbs_mat, flows, param_mat, phase_ids_mat)
		if not np.allclose(Sr, Sr_mat):
			np.testing.assert_allclose(Sr, Sr_mat, equal_nan=True, atol=0.02)
			# save images
			import cv2
			basepath = os.getenv("QBPY_BASE_DIR") + "/testing/results/"
			cv2.imwrite(basepath+"Sr_out_cv2.png", (Sr[:,:,:3] / np.max(Sr)*255).astype(np.uint8))
			cv2.imwrite(basepath+"Sr_mat_out_cv2.png", (np.array(Sr_mat)[:,:,:3] / np.max(Sr_mat) * 255).astype(np.uint8))

	imr = post_merge(Sr.copy(), param, False)
	if eng is not None:
		paramNoBM3D = param_mat.copy()
		paramNoBM3D["bm3dSigma"] = 0
		imr_mat = eng.postMerge(Sr.copy(), paramNoBM3D, False)
		np.testing.assert_allclose(imr.squeeze(), imr_mat, equal_nan=True, atol=0.001)

	result = {
		"ima": ima_py,
		"imas": imas_py,
		"imaf": imaf,
		"imasf": imasf,
		"Sr": Sr,
		"imr": imr,
		"param": param,
		"flows": flows,
		"flowrs": flowrs
	}

	print('Finished merging.')
	return result