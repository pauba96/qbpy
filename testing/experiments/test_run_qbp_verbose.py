import os
import numpy as np
import unittest
import cv2
from testing.io import get_eng
from qbpy.main.run_qbp_verbose import run_qbp_stepwise #  !! verbose version


class TestQbpPipeline(unittest.TestCase):
	def setUp(self):
		"""
		Setup environment and MATLAB engine for tests.
		"""
		self.base = os.getenv("QBPY_BASE_DIR")
		if self.base is None:
			raise EnvironmentError("QBPY_BASE_DIR environment variable not set.")

		self.filepath = os.path.join(
			self.base, "Configs", "multi_ls.json"
		)
		self.filepath_single = os.path.join(
			self.base, "Configs", "single_ls.json"
		)

		# Attempt to initialize MATLAB engine
		try:
			self.eng = get_eng()
		except ImportError:
			self.eng = None
			print("MATLAB engine not available, tests involving MATLAB will be skipped.")

	def test_pipeline_without_matlab(self):
		"""
		Test pipeline execution without MATLAB engine.
		"""
		# Run the Python-only pipeline
		result = run_qbp_stepwise(self.filepath, None)

		# Assert that results are generated
		self.assertIn("Sr", result)
		self.assertIn("imr", result)
		self.assertIn("ima", result)
		self.assertIn("flows", result)
		self.assertIn("flowrs", result)

	def test_pipeline_without_matlab_single_ls(self):
		"""
		Test pipeline execution without MATLAB engine.
		"""
		# Run the Python-only pipeline
		result = run_qbp_stepwise(self.filepath_single, None)

		basepath = os.path.join(os.path.dirname(__file__), "..", "results")
		cv2.imwrite(basepath+"/imr_single.png", (result["imr"][:, :, :3] / np.max(result["imr"]) * 255).astype(np.uint8))

		# Assert that results are generated
		self.assertIn("Sr", result)
		self.assertIn("imr", result)
		self.assertIn("ima", result)
		self.assertIn("flows", result)
		self.assertIn("flowrs", result)

	def test_pipeline_with_matlab(self):
		"""
		Test pipeline execution with MATLAB engine and compare results.
		"""
		if self.eng is None:
			self.skipTest("MATLAB engine not available")

		# Run Python pipeline with MATLAB engine
		result = run_qbp_stepwise(self.filepath, self.eng)

		# Run the MATLAB pipeline
		param_mat = self.eng.param_from_json(self.filepath)
		imbs_mat, dcr_mat, h5_info_mat, dropped_mat, phase_ids_mat = self.eng.load_dataset(param_mat, nargout=5)
		result_matlab = self.eng.qbpPipelineMono(
			imbs_mat, param_mat, dcr_mat, dropped_mat, phase_ids_mat
		)

		# Save images for visual comparison
		basepath = os.path.join(os.path.dirname(__file__), "..", "results/")
		cv2.imwrite(basepath+"imr_out_cv2.png", (result["imr"][:, :, :3] / np.max(result["imr"]) * 255).astype(np.uint8))
		cv2.imwrite(
			basepath+"imr_mat_out_cv2.png",
			(np.array(result_matlab["imr"])[:, :, :3] / np.max(result_matlab["imr"]) * 255).astype(np.uint8),
		)

		# Assertions for equivalence between Python and MATLAB results
		np.testing.assert_allclose(result["Sr"], result_matlab["Sr"], equal_nan=True, atol=1e-10)
		np.testing.assert_allclose(result["imr"], result_matlab["imr"], equal_nan=True, atol=1e-10)
		np.testing.assert_allclose(result["ima"], result_matlab["ima"], equal_nan=True)
		np.testing.assert_allclose(np.array(result["flows"]), result_matlab["flows"], equal_nan=True)
		np.testing.assert_allclose(np.array(result["flowrs"]), result_matlab["flowrs"], equal_nan=True)


if __name__ == "__main__":
	unittest.main()