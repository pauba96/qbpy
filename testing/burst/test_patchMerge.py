import unittest
import numpy as np
from testing.TestFunctions import TestFunctions
from qbpy.burst.patchMerge import patch_merge


class TestPatchMergeFunctions(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for patch_merge tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        patches = data["patches"]  # 4D numpy array of pre-aligned blocks
        param = data["param"]  # Parameter dictionary

        # Prepare MATLAB-compatible parameters
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        # Prepare inputs
        python_inputs = (patches, param)
        matlab_inputs = (np.array(patches).astype(float), param_matlab)

        return python_inputs, matlab_inputs

    def prep_and_run(self, path):
        """
        Test patch_merge with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            patch_merge,
            self.eng.patchMerge,
            python_inputs,
            matlab_inputs,
            rtol=0, atol=1e-3 # Allow for small differences due to floating-point precision
        )

    def test_load_default(self):
        """Test patch_merge with default loaded data."""
        self.prep_and_run("testing/test_data_inputs/patchMerge_inputs.pkl")

    def test_load_error(self):
        """Test patch_merge with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/patch_merge_error.pkl")

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness."""
        H, W, C, N = 64, 64, 4, 5  # Dimensions for patches
        PATCH_SIZE = [16,16,8]
        param = {
            "H": H,
            "W": W,
            "patchSizes": PATCH_SIZE,
            "refImage": 1,
            "wienerC": 8,
            "debug": False,
            "dataType": 'double',
        }

        # Generate synthetic patches
        patches = np.random.rand(H, W, C, N).astype(np.float32)

        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = (np.array(patches).astype(float), param_matlab)

        # Compare results
        self.compare_results(
            patch_merge,
            self.eng.patchMerge,
            python_inputs=(patches, param),
            matlab_inputs=matlab_inputs,
            rtol=0, atol=1e-3  # Allow for small differences due to floating-point
        )

if __name__ == '__main__':
    unittest.main()
