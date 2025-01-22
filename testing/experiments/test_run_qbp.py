import os
import unittest
from testing.io import get_eng
from qbp.main.run_qbp import run_qbp


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
		result = run_qbp(self.filepath)

		# Assert that results are generated
		self.assertIn("Sr", result)
		self.assertIn("imr", result)
		self.assertIn("ima", result)
		self.assertIn("flows", result)
		self.assertIn("flowrs", result)


if __name__ == "__main__":
	unittest.main()