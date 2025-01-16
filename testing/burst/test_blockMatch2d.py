import numpy as np
import unittest
from qbpy.burst.blockMatch2d import block_match_2d_multichannel
from testing.TestFunctions import TestFunctions

class TestBlockMatch(TestFunctions):
    def input_prep(self, path):
        data = self.load_test_data(path)

        # Extract fields of the data
        im0, im1 = data["im0"], data["im1"]
        ylb, xlb = data["block_ul"]
        block_size, searchRadius, initMatch = data["block_size"], data["search_radius"], data["init_match"]

        # Prepare inputs
        python_inputs = (im0, im1, [ylb, xlb], block_size, searchRadius, initMatch)
        matlab_inputs = (im0, im1, np.array([ylb, xlb]).astype(float),
                         float(block_size), float(searchRadius), np.array(initMatch).astype(float))

        return python_inputs,matlab_inputs
    def test_boundary(self):
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep("testing/test_data_inputs/blockMatch_boundary.pkl")
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            block_match_2d_multichannel,
            self.eng.blockMatch2d_multichannel,
            python_inputs,
            matlab_inputs
        )
    def prep_and_run(self, path):
        """
        Test block_match_2d_multichannel with loaded test data. Usually, this loads the initial input to the function.
        Often this can be an all-zero case.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            block_match_2d_multichannel,
            self.eng.blockMatch2d_multichannel,
            python_inputs,
            matlab_inputs
        )

    def test_load_default(self):
        """
        Test block_match_2d_multichannel with loaded test data. In this case, we load the data, that the function last threw an error on.
        """
        self.prep_and_run("testing/test_data_inputs/block_match_2d_multichannel_inputs.pkl")
    def test_load_error(self):
        """
        Test block_match_2d_multichannel with loaded test data. In this case, we load the data, that the function last threw an error on.
        """
        self.prep_and_run("testing/test_data_inputs/block_match_2d_multichannel_error_inputs.pkl")

    def test_load_assertion(self):
        """
        Test block_match_2d_multichannel with loaded test data. In this case, we load the data, that the function last threw an error on.
        """
        self.prep_and_run("testing/test_data_inputs/block_match_2d_multichannel_assertion_inputs.pkl")

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness without MATLAB."""
        IMG_SHAPE = (20, 20, 3)
        SHIFT_U, SHIFT_V = 2, -3
        BLOCK_SIZE = 5
        YLB, XLB = 5, 5
        SEARCH_RADIUS = 5
        INIT_MATCH = [1, 0]

        im0 = np.random.rand(*IMG_SHAPE)
        im1 = np.copy(im0)

        # Introduce a known shift
        ref_patch = im0[YLB:YLB + BLOCK_SIZE, XLB:XLB + BLOCK_SIZE, :]
        shifted_ylb = YLB + SHIFT_V
        shifted_xlb = XLB + SHIFT_U
        im1[shifted_ylb:shifted_ylb + BLOCK_SIZE, shifted_xlb:shifted_xlb + BLOCK_SIZE, :] = ref_patch

        currMatch_py, currScore_py = block_match_2d_multichannel(
            im0, im1, [YLB+1, XLB+1], BLOCK_SIZE, SEARCH_RADIUS, INIT_MATCH, debug=True # +1 to emulate Matlab indexing
        )
        # Check the result
        self.assertEqual(currMatch_py.tolist(), [SHIFT_U, SHIFT_V])

if __name__ == '__main__':
    unittest.main()