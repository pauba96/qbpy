import numpy as np
import unittest
from qbpy.burst.patchAlignBinary import patch_align_binary
from testing.TestFunctions import TestFunctions


class TestPatchAlignBinary(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for patch_align_binary tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        imbs = data["imbs"]  # List of binary frames
        param = data["param"]  # Parameter dictionary

        # Prepare MATLAB-compatible parameters
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        # Prepare inputs
        python_inputs = (imbs, param)
        matlab_inputs = ([np.array(imb).astype(float) for imb in imbs], param_matlab)

        return python_inputs, matlab_inputs

    def prep_and_run(self, path):
        """
        Test patch_align_binary with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            patch_align_binary,
            self.eng.patchAlignBinary,
            python_inputs,
            matlab_inputs
        )

    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-8):
        """
        Override compare_results to handle list outputs for patch_align_binary.
        """
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)

        flows_py, flowrs_py = py_result
        flows_mat, flowrs_mat = matlab_result

        for fm, fp in zip(flows_py, flows_mat):
            np.testing.assert_allclose(fm, fp, equal_nan=True)
        # Compare flows_mat and flows_py
        np.testing.assert_allclose(flowrs_py, flowrs_mat, equal_nan=True)

    def test_load_default(self):
        """Test patch_align_binary with default loaded data."""
        self.prep_and_run("testing/test_data_inputs/patch_align_binary_inputs.pkl")

    def test_load_error(self):
        """Test patch_align_binary with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/patch_align_binary_error.pkl")

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness."""
        IMG_SHAPE = (64, 128)  # Image dimensions
        TW_SIZE = 4  # Temporal window size
        TW_NUM = 3  # Number of temporal windows
        UPSAMPLE_RATIOS = [1, 2, 2]
        PATCH_SIZES = [16, 16, 8]
        SEARCH_RADII = [2, 2, 2]

        # Generate synthetic binary frames
        imbs = [np.random.randint(0, 2, size=IMG_SHAPE).astype(float) for _ in range(TW_SIZE * TW_NUM)]
        param = {
            "alignTWSize": TW_SIZE,
            "alignTWNum": TW_NUM,
            "refFrame": 1,
            "imgScale": 1.0,
            "doRefine": False,
            "resultDir": "./results",
            "debug": False,
            "numLevels": len(UPSAMPLE_RATIOS),
            "patchSizes": PATCH_SIZES,
            "upsampleRatios": UPSAMPLE_RATIOS,
            "searchRadii": SEARCH_RADII,
            "dataType": 'double',
            "numLKIters": 3,
            "fastMode": True,
            "doSR": False
        }

        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = ([np.array(imb).astype(float) for imb in imbs], param_matlab)

        # Compare results
        self.compare_results(
            patch_align_binary,
            self.eng.patchAlignBinary,
            python_inputs=(imbs, param),
            matlab_inputs=matlab_inputs
        )


if __name__ == '__main__':
    unittest.main()
