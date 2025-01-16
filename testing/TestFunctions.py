# tests/test_base.py
import unittest
import numpy as np
from testing.io import get_eng
import os
import matlab
from testing.io import save_inputs, save_error
import pickle
import inspect

def test_logger(func):
    def wrapper(*args, **kwargs):
        if os.getenv("ENABLE_TEST_LOGGER", "0") in ("1", "true", "True"):
            try:
                # Convert args to a dict using the function's signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()  # Fill in default values for missing arguments
                input_dict = bound_args.arguments  # OrderedDict of parameter names and values

                # Save the inputs as a dictionary
                save_inputs(func.__name__, dict(input_dict))
                return func(*args, **kwargs)     # Call the original function
            except Exception as e:
                save_error(func.__name__, args, str(e))  # Save error details
                raise
        else:
            # Skip logging and call the function directly
            return func(*args, **kwargs)
    return wrapper
class TestFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # set ENABLE_TEST_LOGGER=0 to disable logging (we do not want to save inputs for every test)
        os.environ["ENABLE_TEST_LOGGER"] = "0"
        cls.eng = get_eng()  # Initialize MATLAB engine once for the test class
        cls.base = os.getenv("QBPY_BASE_DIR")
        if not cls.base:
            raise EnvironmentError("QBPY_BASE_DIR environment variable not set.")

    @classmethod
    def tearDownClass(cls):
        cls.eng.quit()  # Stop MATLAB engine

    def check_matlab_available(self):
        if self.eng is None:
            self.skipTest("MATLAB engine not available")

    def load_test_data(self, filename):
        """
        Load test data from a file in the specified base directory.

        Args:
            filename: The relative path to the test data file.

        Returns:
            Loaded data (e.g., dictionary or array) depending on the file format.
        """
        data_path = os.path.join(self.base, filename)
        if not os.path.isfile(data_path):
            self.skipTest(f"Test data file not found: {data_path}")

        # Handle .pkl files
        if filename.endswith(".pkl"):
            with open(data_path, "rb") as f:
                return pickle.load(f)

        # Handle .npz files (optional, for flexibility)
        elif filename.endswith(".npz"):
            print(f"Warning: Loading .npz file: {data_path}")
            return np.load(data_path)

        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def run_function(self, python_func, matlab_func, python_inputs, matlab_inputs=None):
        """
        Run the Python and MATLAB functions with the given inputs.

        Args:
            python_func (callable): Python function to execute.
            matlab_func (callable): MATLAB function to execute.
            python_inputs (tuple): Inputs for the Python function.
            matlab_inputs (tuple, optional): Inputs for the MATLAB function. If None, they are derived from python_inputs.

        Returns:
            tuple: (py_result, matlab_result) - Results from Python and MATLAB functions, not normalized to comparable types.
        """
        if matlab_inputs is None:
            # Convert Python inputs to MATLAB-friendly inputs
            matlab_inputs = [matlab.double([x]) if isinstance(x, (int, float)) else x for x in python_inputs]

        # Call Python function
        py_result = python_func(*python_inputs)

        # get number of outputs of py_results (tuple, or single value)
        if isinstance(py_result, tuple):
            nargout = len(py_result)
        else:
            nargout = 1
        # Call MATLAB function
        matlab_result = matlab_func(*matlab_inputs, nargout=nargout)
        return py_result, matlab_result

    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-8):
        """
        Compare results from Python and MATLAB implementations.

        Args:
            python_func: Python function to test.
            matlab_func: MATLAB function to test.
            python_inputs: Inputs for the Python function.
            matlab_inputs: Inputs for the MATLAB function (optional).
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.
        """
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)
        # Convert MATLAB results to Python-friendly format
        if isinstance(matlab_result, tuple):
            matlab_result = tuple(map(lambda x: np.array(x).squeeze(), matlab_result))
        elif isinstance(matlab_result, list):
            matlab_result = [np.array(x).squeeze() for x in matlab_result]
        else:
            matlab_result = np.array(matlab_result).squeeze()
        if isinstance(py_result, tuple):
            # loop over the results and compare them
            for i in range(0, len(py_result)):
                np.testing.assert_allclose(py_result[i], matlab_result[i], rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(py_result, matlab_result, rtol=rtol, atol=atol)
