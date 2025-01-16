import numpy as np
import unittest
from testing.TestFunctions import TestFunctions
from qbpy.burst.buildAgrePyramid import build_aggre_pyramid

class TestBuildAggrePyramid(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for build_aggre_pyramid tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        ims = data["ims"]
        upsample_ratios = data["upsample_ratios"]

        # Prepare inputs
        python_inputs = (ims, upsample_ratios)
        matlab_inputs = (
            np.array(ims).astype(float),  # Convert to MATLAB-compatible array
            np.array(upsample_ratios).astype(float),  # MATLAB expects a list
        )

        return python_inputs, matlab_inputs

    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-8):
        """we override the compare_results method to handle the case where the output is a list of images"""
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)
        for i in range(0, len(py_result)):
            np.testing.assert_allclose(py_result[i], matlab_result[i], rtol=rtol, atol=atol)
    def prep_and_run(self, path):
        """
        Test build_aggre_pyramid with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            build_aggre_pyramid,
            self.eng.buildAggrePyramid,
            python_inputs,
            matlab_inputs
        )

    def test_load_default(self):
        """Test build_aggre_pyramid with default loaded data."""
        self.prep_and_run("testing/test_data_inputs/build_aggre_pyramid_inputs.pkl")

    def test_load_error(self):
        """Test build_aggre_pyramid with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/build_aggre_pyramid_error.pkl")

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness without MATLAB."""
        IMS_SHAPE = (16, 16)  # Image dimensions
        UPSAMPLE_RATIOS = [1, 2, 2]  # Pyramid upsample ratios

        ims = np.random.rand(*IMS_SHAPE)  # Random image for testing
        result_py = build_aggre_pyramid(ims, UPSAMPLE_RATIOS)

        # Verify the pyramid structure
        self.assertEqual(len(result_py), len(UPSAMPLE_RATIOS))
        for i in range(len(result_py)):
            scale_factor = np.prod(UPSAMPLE_RATIOS[:i + 1])
            expected_shape = (IMS_SHAPE[0] // scale_factor, IMS_SHAPE[1] // scale_factor)
            self.assertEqual(result_py[i].shape, expected_shape)

        # Check normalization (values should be <= 1)
        for level in result_py:
            self.assertTrue(np.all(level <= 1), f"Values exceed 1 at level {level}")

if __name__ == '__main__':
    unittest.main()
