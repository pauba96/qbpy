import os
import numpy as np
import unittest
import cv2
from testing.io import get_eng
from qbp.main.run_qbp_verbose import run_qbp_stepwise_verbose #  !! verbose version
from qbp.main.run_qbp import run_qbp


def save_image(image, filepath):
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Single-channel (grayscale)
        cv2.imwrite(filepath, (image / np.max(image) * 255).astype(np.uint8))
    elif image.ndim == 3 and image.shape[2] >= 3:
        # Multi-channel (RGB or more)
        cv2.imwrite(filepath, (image[:, :, :3] / np.max(image) * 255).astype(np.uint8))
    else:
        raise ValueError("Unexpected image shape: {}".format(image.shape))
class TestQbpPipeline(unittest.TestCase):
	def setUp(self):
		"""
		Setup environment and MATLAB engine for tests.
		"""
		self.base = os.getenv("QBPY_BASE_DIR")
		if self.base is None:
			raise EnvironmentError("QBPY_BASE_DIR environment variable not set.")

		self.filepath = os.path.join(
			self.base, "Configs", "single_ls.json"
		)
		self.filepath_multi = os.path.join(
			self.base, "Configs", "multi_ls.json"
		)

		# Attempt to initialize MATLAB engine
		try:
			self.eng = get_eng()
		except ImportError:
			self.eng = None
			print("MATLAB engine not available, tests involving MATLAB will be skipped.")
	def test_pipeline_with_UW_examples(self):
		"""
		Test pipeline execution with MATLAB engine and compare results.
		"""
		filepath = os.path.join(
			self.base, "Configs", "single_UW_example.json"
		)
		if self.eng is None:
			self.skipTest("MATLAB engine not available")

		# Run Python pipeline with MATLAB engine
		result = run_qbp_stepwise_verbose(filepath, self.eng)

		# Run the MATLAB pipeline
		param_mat = self.eng.param_from_json(filepath)
		imbs_mat, dcr_mat, h5_info_mat, dropped_mat, phase_ids_mat = self.eng.load_dataset(param_mat, nargout=5)
		result_matlab = self.eng.qbpPipelineMono(
			imbs_mat, param_mat, dcr_mat, dropped_mat, phase_ids_mat
		)

		# Save images for visual comparison
		basepath = os.path.join(os.path.dirname(__file__), "..", "results/")

		save_image(np.clip(result["imr"],0,1), basepath + "/imr_out_verbose.png")
		save_image(np.clip(np.array(result_matlab["imr"]),0,1), basepath + "/imr_mat_out_verbose.png")

		# Assertions for equivalence between Python and MATLAB results
		np.testing.assert_allclose(result["Sr"], result_matlab["Sr"], equal_nan=True, atol=1e-10)
		np.testing.assert_allclose(result["imr"].squeeze(), result_matlab["imr"], equal_nan=True, atol=1e-10)
		np.testing.assert_allclose(result["ima"], result_matlab["ima"], equal_nan=True)
		np.testing.assert_allclose(np.array(result["flows"]), result_matlab["flows"], equal_nan=True)
		np.testing.assert_allclose(np.array(result["flowrs"]), result_matlab["flowrs"], equal_nan=True)

	def test_pipeline_with_UW_examples_no_matlab(self):
		"""
		Test pipeline execution with MATLAB engine and compare results.
		"""
		filepath = os.path.join(
			self.base, "Configs", "single_UW_example.json"
		)
		# Run Python pipeline with MATLAB engine
		result = run_qbp(filepath)
		# Save images for visual comparison
		basepath = os.path.join(os.path.dirname(__file__), "..", "results/")
		save_image(np.clip(result["imr"],0,1), basepath + "/imr_out_nomat.png")
		save_image(np.clip(result["ima"],0,1), basepath + "/ima_out_nomat.png")

if __name__ == "__main__":
	unittest.main()