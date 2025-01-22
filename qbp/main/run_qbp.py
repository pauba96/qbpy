import os
import numpy as np
from qbp.utils.ps_shape.param_from_json import param_from_json
from qbp.utils.ps_shape.load_dataset import load_dataset
from qbp.burst.naiveRecons import naive_recons
from qbp.burst.patchAlignBinary import patch_align_binary
from qbp.burst.patchMergeBinary import patch_merge_binary
from qbp.burst.postMerge import post_merge

def run_qbp(json_path):
	param = param_from_json(json_path)
	# Check if the parameters are equivalent
	# run python version of load_dataset and check if equal
	imbs_py, dcr_py, h5_info_py, dropped_py, phase_ids_py = load_dataset(param)


	# Naive reconstruction
	ima_py, S = naive_recons(imbs_py, param)

	# reference block naive reconstruction
	asParam_py = param.copy()
	asParam_py["mergeTWNum"] = 1
	refBlock = np.floor((param["refFrame"] - 1) / param["mergeTWSize"])
	start_idx = int(refBlock * param["mergeTWSize"])
	stop_idx = int((refBlock + 1) * param["mergeTWSize"])
	sliced_imbs_py = imbs_py[start_idx:stop_idx]
	imas_py, S = naive_recons(sliced_imbs_py, asParam_py)

	# Remove hot pixels
	if param["removeHP"]:
		raise NotImplementedError("removeHP is not implemented in python")
	else:
		imaf = []
		imasf = []
	print('Finished naive reconstruction.')

	# Align
	flows, flowrs = patch_align_binary(imbs_py, param)


	# Merge
	Sr = patch_merge_binary(imbs_py, flows, param, phase_ids_py)

	# Convert photon counts back to intensity
	imr = post_merge(Sr, param, False)

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

if __name__ == "__main__":
	base = os.getenv("QBPY_BASE_DIR")
	filepath = os.path.join(
		base, "Configs", "single_ls.json"
	)
	result = run_qbp(filepath)
