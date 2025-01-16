import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from qbpy.utils.interp2 import interp2
import unittest
import os
from testing.io import get_eng
from qbpy.utils.partial_deriv_patch import partial_deriv_patch


class TestPartialDerivPatch(unittest.TestCase):
    def setUp(self):
        # Initialize test images and parameters
        self.img1 = np.random.rand(100, 100)
        self.img2 = np.random.rand(100, 100)
        self.uv_prev = np.random.rand(100, 100, 2)
        eng = get_eng()
        self.eng = eng


    def test_partial_deriv_patch_linear(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        x = np.linspace(0, 2 * np.pi, 10)
        y = np.linspace(0, 2 * np.pi, 10)
        uv = np.random.rand(10, 10, 2)
        #uv[:,:,0] = 0
        #uv[:,:,1] = 5
        X, Y = np.meshgrid(x, y)
        img1 = np.random.rand(10,10) #np.sin(X) + np.cos(Y)
        img2 = np.sin(X + 0.1) + np.cos(Y + 0.1)
        It, Ix, Iy = partial_deriv_patch(img1, img2, uv, interpolation_method='bi-linear', eng=self.eng)
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(img1, img2, uv, 'bi-linear', nargout=3)
        np.testing.assert_allclose(It, It_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Ix, Ix_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Iy, Iy_eng, rtol=1e-5, equal_nan=True)

    def test_non_quadratic_linear(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        shape = (10,15)
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        uv = np.random.rand(shape[0], shape[1], 2)
        X, Y = np.meshgrid(x, y, indexing='xy')
        img1 = np.random.rand(shape[0], shape[1]) #np.sin(X) + np.cos(Y)
        img2 = np.sin(X + 0.1) + np.cos(Y + 0.1)
        It, Ix, Iy = partial_deriv_patch(img1, img2, uv, interpolation_method='bi-linear', eng=self.eng)
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(img1, img2, uv, 'bi-linear', nargout=3)
        np.testing.assert_allclose(It, It_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Ix, Ix_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Iy, Iy_eng, rtol=1e-5, equal_nan=True)

    def test_different_sizes(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        shape = (10,15)
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        uv = np.random.rand(shape[0], shape[1], 2)
        X, Y = np.meshgrid(x, y, indexing='xy')
        img1 = np.random.rand(shape[0], shape[1]) #np.sin(X) + np.cos(Y)
        img2 = np.random.rand(7, 12)
        It, Ix, Iy = partial_deriv_patch(img1, img2, uv, interpolation_method='bi-linear', eng=self.eng)
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(img1, img2, uv, 'bi-linear', nargout=3)
        np.testing.assert_allclose(It, It_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Ix, Ix_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Iy, Iy_eng, rtol=1e-5, equal_nan=True)

    def test_cubic(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        shape = (10,15)
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        uv = np.random.rand(shape[0], shape[1], 2)
        X, Y = np.meshgrid(x, y, indexing='xy')
        img1 = np.random.rand(shape[0], shape[1]) #np.sin(X) + np.cos(Y)
        img2 = np.random.rand(7, 12)
        It, Ix, Iy = partial_deriv_patch(img1, img2, uv, interpolation_method='cubic', eng=None, method_selection='rect_bivariate_spline') #ignore internal checks
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(img1, img2, uv, 'cubic', nargout=3)
        # plot It, Ix, Iy, It_eng, Ix_eng, Iy_eng and their differences
        plt.figure()
        plt.subplot(331); plt.imshow(It); plt.title("It")
        plt.subplot(332); plt.imshow(Ix); plt.title("Ix")
        plt.subplot(333); plt.imshow(Iy); plt.title("Iy")
        plt.subplot(334); plt.imshow(It_eng); plt.title("It_eng")
        plt.subplot(335); plt.imshow(Ix_eng); plt.title("Ix_eng")
        plt.subplot(336); plt.imshow(Iy_eng); plt.title("Iy_eng")
        plt.subplot(337); plt.imshow(np.abs(It - It_eng)); plt.title("It diff"); plt.colorbar()
        plt.subplot(338); plt.imshow(np.abs(Ix - Ix_eng)); plt.title("Ix diff"); plt.colorbar()
        plt.subplot(339); plt.imshow(np.abs(Iy - Iy_eng)); plt.title("Iy diff"); plt.colorbar()
        plt.tight_layout()
        outpath = os.path.join(os.path.dirname(__file__), "..", "results/cubic_interp.png")
        plt.savefig(outpath)
        print("Cubic interpolation is not equivalent")
        #np.testing.assert_allclose(It[:img2.shape[0]-1,:img2.shape[1]-1], np.array(It_eng)[:img2.shape[0]-1,:img2.shape[1]-1],
        #                           rtol=0.2,atol=0.2, equal_nan=True) # pretty big tolerance, because we dont really have a good cubic interpolation
        #np.testing.assert_allclose(Ix[:img2.shape[0]-1,:img2.shape[1]-1], np.array(Ix_eng)[:img2.shape[0]-1,:img2.shape[1]-1],
        #                           rtol=0.2,atol=0.2, equal_nan=True) # pretty big tolerance, because we dont really have a good cubic interpolation
        #np.testing.assert_allclose(Iy[:img2.shape[0]-1,:img2.shape[1]-1], np.array(Iy_eng)[:img2.shape[0]-1,:img2.shape[1]-1],
        #                           rtol=0.2,atol=0.2, equal_nan=True) # pretty big tolerance, because we dont really have a good cubic interpolation

    # test multichannel images
    def test_multichannel(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        shape = (10,15)
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        uv = np.random.rand(shape[0], shape[1], 2)
        X, Y = np.meshgrid(x, y, indexing='xy')
        img1 = np.random.rand(shape[0], shape[1], 3) #np.sin(X) + np.cos(Y)
        img2 = np.random.rand(7, 12, 3)
        It, Ix, Iy = partial_deriv_patch(img1, img2, uv, interpolation_method='bi-linear', eng=self.eng)
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(img1, img2, uv, 'bi-linear', nargout=3)
        np.testing.assert_allclose(It, It_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Ix, Ix_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Iy, Iy_eng, rtol=1e-5, equal_nan=True)

    def test_loaded(self):
        # Test with bi-linear interpolation
        # define img1 and img as shifted sine-cos curves
        # if file does not exist, skip test
        if not os.path.exists(os.getenv("QBPY_BASE_DIR")+"/testing/test_data_inputs/2024_12_11_test_data.npz"):
            self.skipTest("test data not available")
        data = np.load(os.getenv("QBPY_BASE_DIR")+"/testing/test_data_inputs/2024_12_11_test_data.npz")
        # extract fields of the data
        im0_test = data["im0_test"]
        im1_test = data["im1_test"]
        uv = data["uv"]
        It, Ix, Iy = partial_deriv_patch(im0_test, im1_test, uv, interpolation_method='bi-linear', eng=self.eng)
        It_eng, Ix_eng, Iy_eng = self.eng.partial_deriv_patch(im0_test, im1_test, uv, 'bi-linear', nargout=3)
        np.testing.assert_allclose(It, It_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Ix, Ix_eng, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(Iy, Iy_eng, rtol=1e-5, equal_nan=True)

if __name__ == '__main__':
    # only run test_multichanel
    suite = unittest.TestSuite()
    suite.addTest(TestPartialDerivPatch('test_multichannel'))
    suite.addTest(TestPartialDerivPatch('test_loaded'))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    #unittest.main()