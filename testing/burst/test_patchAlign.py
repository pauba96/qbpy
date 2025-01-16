import numpy as np
import unittest
from testing.TestFunctions import TestFunctions
from qbpy.burst.patchAlign import patch_align

class TestPatchAlign(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for patch_align tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        ims = data["ims"]  # List of images
        param = data["param"]  # Parameter dictionary
        imv = data["imv"]
        # Prepare inputs
        python_inputs = (ims, param, imv)

        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = (
            [np.array(im).astype(float) for im in ims],  # Convert each image
            param_matlab,
            [np.array(im).astype(float) for im in imv]  # MATLAB-friendly dictionary
        )

        return python_inputs, matlab_inputs

    def prep_and_run(self, path):
        """
        Test patch_align with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            patch_align,
            self.eng.patchAlign,
            python_inputs,
            matlab_inputs
        )

    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-8):
        """
        Override compare_results to handle list outputs for patch_align.
        """
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)

        # Ensure outputs are comparable
        for py_flow, matlab_flow in zip(py_result, matlab_result):
            if py_flow is not None and matlab_flow is not None:
                np.testing.assert_allclose(py_flow, matlab_flow, rtol=rtol, atol=atol, equal_nan=True)

    def test_load_default(self):
        """Test patch_align with default loaded data."""
        print("skip")
        #self.prep_and_run("testing/test_data_inputs/patch_align_inputs.pkl")

    def test_load_error(self):
        """Test patch_align with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/patch_align_error.pkl")

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness."""
        IMG_SHAPE = (64, 128)  # Image dimensions
        N_IMAGES = 5  # Number of images
        UPSAMPLE_RATIOS = [1, 2, 2]  # Pyramid upsample ratios
        PATCH_SIZES = [16, 16, 8]  # Patch sizes for each level -> must be even numbers
        SEARCH_RADII = [2, 2, 2]  # Search radii for each level

        # Generate synthetic images
        ims = [np.random.randint(0, 2, size=IMG_SHAPE).astype(float) for _ in range(N_IMAGES)]
        imv = [np.random.rand(*IMG_SHAPE).astype(float) for _ in range(N_IMAGES)]
        param = {
            "refImage": 2,
            "upsampleRatios": UPSAMPLE_RATIOS,
            "numLevels": len(UPSAMPLE_RATIOS),
            "patchSizes": PATCH_SIZES,
            "searchRadii": SEARCH_RADII,
            "resultDir": "./results",
            "debug": False,
            "dataType": 'double',
            "numLKIters": 3,
            "fastMode": True,
            "doSR": False
        }

        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = (
            [np.array(im).astype(float) for im in ims],  # Convert each image
            param_matlab,
            [np.array(im).astype(float) for im in imv]  # MATLAB-friendly dictionary
        )

        # Compare results
        self.compare_results(
            patch_align,
            self.eng.patchAlign,
            python_inputs=(ims, param, imv),
            matlab_inputs=matlab_inputs
        )

if __name__ == '__main__':
    unittest.main()
