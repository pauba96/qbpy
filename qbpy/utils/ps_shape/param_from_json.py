import json
import os


def envsubst(value):
	"""Replace environment variables in the given string."""
	return os.path.expandvars(value)


def param_from_json(json_file_path):
	"""Load parameters from a JSON file and merge them with default parameters."""
	data_dir = "not set, provide in json"
	dcr_path = "not set, provide in json"
	result_dir = "not set, provide in json"

	default_param = {
		'dataDir': data_dir,
		'dataset_type': 'qbp',
		'target_size': [512, 512],
		'alignTWSize': 100, 'alignTWNum': 20,
		'mergeTWSize': 100, 'mergeTWNum': 20, 'warpTWSize': 10,
		'srTWSize': 40, 'srTWNum': 50,
		'refFrame': calc_ref_frame(100, 20),
		'numLevels': 3, 'patchSizes': [16, 16, 8],
		'upsampleRatios': [1, 2, 4], 'searchRadii': [1, 4, 8], 'numLKIters': 3,
		'imgScale': 1, 'imgAutoScale': True,
		'wienerC': 8,
		'flowLambda': 0.01,
		'srScale': 2, 'combineRadius': 1,
		'k_detail': 0.3, 'k_denoise': 1, 'D_th': 0.005, 'D_tr': 0.5, 'k_stretch': 1, 'k_shrink': 1,
		'wienerSRC': 8,
		'bm3dSigma': 0,
		'hpThresh': 50, 'correctDCR': False, 'removeHP': True,
		'dcrPath': dcr_path,
		'fastMode': True, 'dataType': 'double',
		'doRefine': False, 'doSR': False, 'doRefineSR': False,
		'computePSNR': False,
		'debug': False, 'saveImages': True, 'resultDir': result_dir,
		'image_path': data_dir, 'use_gt_flow': "not defined",
		'calculate_flow': True, 'debug_flow': True, 'perFrameGt': False,
		'loop_lvl': 0
	}

	if json_file_path:
		if os.path.isfile(json_file_path):
			with open(json_file_path, 'r') as f:
				json_data = json.load(f)

			if 'qbp' in json_data:
				qbp_data = json_data['qbp']
				for key, value in qbp_data.items():
					if isinstance(value, str):
						value = envsubst(value)
					elif isinstance(value, list):
						value = [envsubst(v) if isinstance(v, str) else v for v in value]
					default_param[key] = value
			else:
				raise ValueError('Specified JSON file does not contain "qbp" field.')

			if 'dataset' in json_data:
				default_param['n_binary'] = json_data['dataset'].get('n_binary', 1)
			else:
				default_param['n_binary'] = 1
		else:
			raise FileNotFoundError('Specified JSON file does not exist.')

	return default_param


def calc_ref_frame(merge_tw_size, merge_tw_num):
	"""Calculate the reference frame based on mergeTWSize and mergeTWNum."""
	return (merge_tw_size * merge_tw_num) // 2