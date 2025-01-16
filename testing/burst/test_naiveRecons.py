import numpy as np
import unittest
from testing.TestFunctions import TestFunctions
from qbpy.burst.naiveRecons import naive_recons

class TestNaiveRecons(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for naive_recons tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        imbs = data["imbs"]  # List of binary frames
        param = data["param"]  # Parameter dictionary

        # Prepare inputs
        python_inputs = (imbs, param)
        matlab_inputs = (
            [np.array(imb).astype(float) for imb in imbs],  # Convert each frame
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}  # MATLAB-friendly dictionary
        )

        return python_inputs, matlab_inputs

    def prep_and_run(self, path):
        """
        Test naive_recons with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            naive_recons,
            self.eng.naiveRecons,
            python_inputs,
            matlab_inputs
        )
    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-8):
        """
        we override the compare_results method so we can handle the second output with shape (h,w,1) in python and (h,w) in matlab
        """
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)
        np.testing.assert_allclose(py_result[0], matlab_result[0], rtol=rtol, atol=atol, equal_nan=True)
        np.testing.assert_allclose(py_result[1].squeeze(), matlab_result[1], rtol=rtol, atol=atol, equal_nan=True)
    def test_load_default(self):
        """Test naive_recons with default loaded data."""
        self.prep_and_run("testing/test_data_inputs/naive_recons_inputs.pkl")

    def test_load_error(self):
        """Test naive_recons with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/naive_recons_error.pkl")

    def test_synthetic_multichannel(self):
        """Create a small synthetic test to verify correctness without MATLAB."""
        IMG_SHAPE = (16, 16, 3)  # Image dimensions
        TW_SIZE = 2  # Temporal window size
        TW_NUM = 3  # Number of temporal windows
        IMG_SCALE = 1  # Scaling factor
        N_BINARY = 1  # Binary intensity normalization factor

        # Generate synthetic binary frames
        imbs = [np.random.randint(0, 2, size=IMG_SHAPE).astype(float) for _ in range(TW_SIZE * TW_NUM)]
        param = {
            "mergeTWSize": TW_SIZE,
            "mergeTWNum": TW_NUM,
            "refFrame": 1,
            "imgScale": IMG_SCALE,
            "n_binary": N_BINARY,
            "dataType": 'double',
        }

        matlab_inputs = (
            [np.array(imb).astype(float) for imb in imbs],  # Convert each frame
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}  # MATLAB-friendly dictionary
        )
        # matlab version
        self.compare_results(
            naive_recons,
            self.eng.naiveRecons,
            python_inputs=[imbs, param],
            matlab_inputs=matlab_inputs
        )

if __name__ == '__main__':
    unittest.main()
