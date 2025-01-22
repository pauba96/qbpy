import unittest
import numpy as np
import os
from qbp.burst.patchMergeBinary import patch_merge_binary
from qbp.burst.patchAlign_subfuns.dc_utils import save_to_mat
from testing.TestFunctions import TestFunctions


class TestPatchMergeBinary(TestFunctions):
    def input_prep(self, path):
        """
        Prepare inputs for patch_merge_binary tests.
        Args:
            path (str): Path to the test data file.
        Returns:
            tuple: Python and MATLAB-compatible inputs.
        """
        data = self.load_test_data(path)

        # Extract fields of the data
        imbs = data["imbs"]
        flows = data["flows"]
        param = data["param"]
        phase_ids = data["phase_ids"]

        # Prepare MATLAB-compatible inputs
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = (
            [np.array(im).astype(float) for im in imbs],
            [np.array(flow).astype(float) for flow in flows],
            param_matlab,
            np.array(phase_ids).astype(float),
        )

        python_inputs = (imbs, flows, param, phase_ids)

        return python_inputs, matlab_inputs

    def prep_and_run(self, path):
        """
        Test patch_merge_binary with loaded test data.
        """
        self.check_matlab_available()
        try:
            python_inputs, matlab_inputs = self.input_prep(path)
        except unittest.SkipTest as e:
            self.skipTest(str(e))

        # Compare results using the base class method
        self.compare_results(
            patch_merge_binary,
            self.eng.patchMergeBinary,
            python_inputs,
            matlab_inputs
        )

    def compare_results(self, python_func, matlab_func, python_inputs, matlab_inputs=None, rtol=1e-5, atol=1e-5):
        """
        Override compare_results to handle matrix outputs for patch_merge_binary.
        """
        py_result, matlab_result = self.run_function(python_func, matlab_func, python_inputs, matlab_inputs)

        # Compare the merged outputs
        np.testing.assert_allclose(
            py_result, matlab_result, rtol=rtol, atol=atol, equal_nan=True,
            err_msg="Mismatch between Python and MATLAB results."
        )

    def test_load_default(self):
        """Test patch_merge_binary with default loaded data."""
        self.prep_and_run("testing/test_data_inputs/patch_merge_binary_inputs.pkl")

    def test_load_error(self):
        """Test patch_merge_binary with the last error-causing data."""
        self.prep_and_run("testing/test_data_inputs/patch_merge_binary_error.pkl")

    def test_synthetic_single_twnum(self):
        """Create a small synthetic test to verify correctness."""
        H, W = 64, 64  # Image dimensions
        N_FRAMES = 8  # Number of frames
        PHASE_IDS = [0, 1, 2, 3, 0, 1, 2, 3]  # Example phase IDs
        PATCH_SIZE = [16,16,8]  # Patch size

        # Generate synthetic data
        imbs = [np.random.rand(H, W).astype(np.float32) for _ in range(N_FRAMES)]
        flows = [np.zeros((H, W, 2), dtype=np.float32) for _ in range(N_FRAMES)]
        param = {
            "alignTWSize": 4,
            "alignTWNum": 1,
            "mergeTWSize": 4,
            "mergeTWNum": 1,
            "warpTWSize": 4,
            "refFrame": 1,
            "patchSizes": np.array(PATCH_SIZE),
            "dataType": 'double',
            "imgScale": 1.0,
            "num_ls": 3,
            "fastMode": False,
            "H": H,
            "W": W
        }

        # Prepare MATLAB-compatible inputs
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}

        matlab_inputs = (
            [np.array(im).astype(float) for im in imbs],
            [np.array(flow).astype(float) for flow in flows],
            param_matlab,
            np.array(PHASE_IDS).astype(float),
        )

        # save synthetic data as matlab
        #save_to_mat(self.eng, "../../test_data/patch_merge_binary_synthetic.mat", imbs=[np.array(im).astype(float) for im in imbs],
        #            flows=[np.array(flow).astype(float) for flow in flows],
        #            param=param_matlab,
        #            phase_ids=np.array(PHASE_IDS).astype(float))

        # Compare results
        self.compare_results(
            patch_merge_binary,
            self.eng.patchMergeBinary,
            python_inputs=(imbs, flows, param, PHASE_IDS),
            matlab_inputs=matlab_inputs
        )

    def test_synthetic(self):
        """Create a small synthetic test to verify correctness."""
        H, W = 64, 32  # Image dimensions
        N_FRAMES = 23  # Number of frames
        # phase ids: 0,1,2,3,0,1,2,3, ... (N_FRAMES times)
        PHASE_IDS = [0, 1, 2, 3] * (N_FRAMES // 4)
        PATCH_SIZE = [16,16,8]  # Patch size
        PATCH_STRIDE = 8
        TWNUM = 5
        patchStride = PATCH_SIZE[0] / 2
        hs = (H - PATCH_SIZE[0]) / patchStride + 1 # 63 for 512x512 images
        ws = (W - PATCH_SIZE[0]) / patchStride + 1
        # assert hs and ws are integers
        assert hs % 1 == 0 and ws % 1 == 0
        # Generate synthetic data
        imbs = [np.random.randint(0, 2, (H, W)).astype(np.float32) for _ in range(N_FRAMES)]
        # flows should be random integers between -100 and 100
        flows = [np.random.randint(-int(H/20), int(H/20), (int(hs), int(ws), 2)).astype(np.float32) for _ in range(TWNUM)]
        param = {
            "alignTWSize": 4,
            "alignTWNum": TWNUM,
            "mergeTWSize": 4,
            "mergeTWNum": TWNUM,
            "refFrame": 12,
            "patchSizes": np.array(PATCH_SIZE).astype(float),
            "dataType": 'double',
            "imgScale": 1.0,
            "num_ls": 4,
            "fastMode": False,
            "H": H,
            "W": W,
            "debug": True,
            "wienerC": 8,
            "upsampleRatios": np.array([1.0, 2.0, 4.0]),
            "searchRadii": np.array([1.0,4.0,16.0]),
            "resultDir": "test_data"
        }
        # Prepare MATLAB-compatible inputs
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}
        matlab_inputs = (
            [np.array(im).astype(float) for im in imbs],
            [np.array(flow).astype(float) for flow in flows],
            param_matlab,
            np.array(PHASE_IDS).astype(float),
        )
        # save synthetic data as matlab
        #save_to_mat(self.eng, os.getenv("QBPY_BASE_DIR") + r"/testing/test_data_inputs/patch_merge_binary_synthetic.mat", imbs=[np.array(im).astype(float) for im in imbs],
        #            flows=[np.array(flow).astype(float) for flow in flows],
        #            param=param_matlab,
        #            phase_ids=np.array(PHASE_IDS).astype(float))
        # Compare results
        self.compare_results(
            patch_merge_binary,
            self.eng.patchMergeBinary,
            python_inputs=(imbs, flows, param, PHASE_IDS),
            matlab_inputs=matlab_inputs
        )

    def test_synthetic_warp(self):
        """Create a small synthetic test to verify correctness."""
        H, W = 64, 32  # Image dimensions
        N_FRAMES = 23  # Number of frames
        PHASE_IDS = False # !! no Phase ids -> match with channels
        PATCH_SIZE = [16,16,8]  # Patch size
        PATCH_STRIDE = 8
        TWNUM = 5
        patchStride = PATCH_SIZE[0] / 2
        hs = (H - PATCH_SIZE[0]) / patchStride + 1 # 63 for 512x512 images
        ws = (W - PATCH_SIZE[0]) / patchStride + 1
        # assert hs and ws are integers
        assert hs % 1 == 0 and ws % 1 == 0
        # Generate synthetic data
        imbs = [np.random.randint(0, 2, (H, W)).astype(np.float32) for _ in range(N_FRAMES)]
        # flows should be random integers between -100 and 100
        flows = [np.random.randint(-int(H/20), int(H/20), (int(hs), int(ws), 2)).astype(np.float32) for _ in range(TWNUM)]
        param = {
            "alignTWSize": 4,
            "alignTWNum": TWNUM,
            "mergeTWSize": 4,
            "mergeTWNum": TWNUM,
            "warpTWSize": 4, # !!
            "refFrame": 12,
            "patchSizes": np.array(PATCH_SIZE).astype(float),
            "dataType": 'double',
            "imgScale": 1.0,
            "num_ls": 4,
            "fastMode": False,
            "H": H,
            "W": W,
            "debug": True,
            "wienerC": 8,
            "upsampleRatios": np.array([1.0, 2.0, 4.0]),
            "searchRadii": np.array([1.0,4.0,16.0]),
            "resultDir": "test_data"
        }
        # Prepare MATLAB-compatible inputs
        param_matlab = {k: float(v) if isinstance(v, (int, float)) else v for k, v in param.items()}
        param_matlab = {k: np.array(v).astype(float) if isinstance(v, list) else v for k, v in param_matlab.items()}
        matlab_inputs = (
            [np.array(im).astype(float) for im in imbs],
            [np.array(flow).astype(float) for flow in flows],
            param_matlab,
            np.array(PHASE_IDS).astype(float),
        )
        # save synthetic data as matlab
        save_to_mat(self.eng, os.getenv("QBPY_BASE_DIR") + r"/testing/test_data_inputs/patch_merge_binary_synthetic_warp.mat", imbs=[np.array(im).astype(float) for im in imbs],
                    flows=[np.array(flow).astype(float) for flow in flows],
                    param=param_matlab,
                    phase_ids=np.array(PHASE_IDS).astype(float))
        # Compare results
        self.compare_results(
            patch_merge_binary,
            self.eng.patchMergeBinary,
            python_inputs=(imbs, flows, param, PHASE_IDS),
            matlab_inputs=matlab_inputs
        )

if __name__ == '__main__':
    unittest.main()
