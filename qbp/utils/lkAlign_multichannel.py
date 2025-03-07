import numpy as np
import unittest
import os
from qbp.utils.ps_shape.param_from_json import param_from_json
from qbp.burst.buildAgrePyramid import build_aggre_pyramid
from qbp.utils.partial_deriv_patch import partial_deriv_patch
from testing.io import get_eng
from testing.TestFunctions import test_logger


@test_logger
def lk_align_multichannel(im0, im1, iters, uv0=None):
    """
    Align two images (2D translation) using Lucas-Kanade for both single and multi-channel images.

    Args:
        im0 (numpy array): Reference image.
        im1 (numpy array): Image to be aligned.
        iters (int): Number of iterations.
        uv0 (numpy array, optional): Initial displacement. Default is None.

    Returns:
        numpy array: Displacement vector.
    """
    if uv0 is None:
        uv = np.zeros((1, 1, 2), dtype=im0.dtype)
    else:
        uv = uv0.reshape((1, 1, 2))

    for _ in range(iters):
        It = np.zeros_like(im0[..., 0])
        Ix = np.zeros_like(im0[..., 0])
        Iy = np.zeros_like(im0[..., 0])

        for c in range(im0.shape[2]):
            It_, Ix_, Iy_ = partial_deriv_patch(im0[..., c], im1[..., c], uv)
            It += It_
            Ix += Ix_
            Iy += Iy_

        A = np.stack((Ix.ravel(), Iy.ravel()), axis=1)
        b = -It.ravel()

        if np.linalg.matrix_rank(A) < 2:
            break

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if np.linalg.norm(x) > 1:
            x = x / np.linalg.norm(x)

        uv = uv + x.reshape(uv.shape)

    return uv.reshape((1, 2))



class TestLKAlign(unittest.TestCase):
    def setUp(self):
        # Setup: Load environment and MATLAB engine if available
        self.base = os.getenv("SPAD3D_BASE")
        if self.base is None:
            raise EnvironmentError("SPAD3D_BASE not set.")
        try:
            import matlab.engine
            self.eng = get_eng()
        except ImportError:
            self.eng = None
            print("MATLAB engine not available, tests involving MATLAB will be skipped.")

        # Load test data
        # Replace with your actual test data path
        data_path = os.path.join(self.base, "testing/test_data_inputs/2024_12_11_patchAlign.npz")
        self.assertTrue(os.path.isfile(data_path), "Test data file not found.")
        self.data = np.load(data_path, allow_pickle=True)

        # Load parameters from json
        # Replace with your actual parameters json
        json_path = os.path.join(self.base, r"tests/21_frame_offset/settings_20241112_150340_speed_800_twsize_4.json")
        self.param = param_from_json(json_path)

        # Build pyramids for test images similar to previous tests
        blockAggres_raw = self.data["blockAggres"]
        # Create a list of images (H,W) and ensure they have a channel dimension
        self.ims = [blockAggres_raw[i, :, :] for i in range(blockAggres_raw.shape[0])]
        self.ims = [im[:, :, None] for im in self.ims]
        self.param["refImage"] = self.data["refBlock"]
        refImgIndex = self.param['refImage']
        self.P0 = build_aggre_pyramid(self.ims[refImgIndex], self.param['upsampleRatios'])

        # If we have at least 2 images, use the second as target
        if len(self.ims) > 1:
            self.P1 = build_aggre_pyramid(self.ims[1], self.param['upsampleRatios'])
        else:
            # Otherwise, duplicate the reference
            self.P1 = build_aggre_pyramid(self.ims[0], self.param['upsampleRatios'])

    def test_lk_align(self):
        if self.eng is None:
            self.skipTest("MATLAB engine not available")

        # Test lk_align at the coarsest level for simplicity
        l = self.param['numLevels'] - 1
        refImg = np.dstack([self.P0[l]]*3)  # Take the single channel
        tgtImg = np.dstack([self.P1[l]]*3)

        #refImg = np.dstack(self.P0[l])
        #tgtImg = np.dstack(self.P1[l])

        # Set iterations
        iters = self.param.get('numLKIters', 3)
        # Initial displacement
        uv0 = np.zeros((1, 2), dtype=refImg.dtype)

        # Call Python lk_align
        uv_py = lk_align_multichannel(refImg, tgtImg, iters, uv0=uv0)

        # Call MATLAB lk_align
        refImg_mat = self.eng.double(refImg)
        tgtImg_mat = self.eng.double(tgtImg)
        uv0_mat = self.eng.double(np.zeros((1, 2), dtype=refImg.dtype))

        # MATLAB version of lk_align must be available:
        uv_mat = self.eng.lkAlign(refImg_mat, tgtImg_mat, float(iters), uv0_mat, nargout=1)

        # Compare results
        np.testing.assert_allclose(uv_py, uv_mat, atol=1e-6, equal_nan=True)



if __name__ == '__main__':
    unittest.main()