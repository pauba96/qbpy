import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from qbp.utils.interp2 import interp2
from testing.TestFunctions import test_logger


@test_logger
def partial_deriv_patch(img1, img2, uv_prev, interpolation_method='bi-linear', deriv_filter=None, b=0.5, eng=None, method_selection='regular_grid'):
    """
    Compute spatio-temporal derivatives for aligning a template image img1 to img2.

    Parameters:
        img1 (ndarray): The first image (template).
        img2 (ndarray): The second image (target).
        uv_prev (ndarray): The optical flow (2D displacement field). Given in y,x order
        interpolation_method (str): Interpolation method ('cubic' or 'linear').
        deriv_filter (ndarray): Derivative filter. Default is used if None.
        b (float): Blending ratio.

    Returns:
        It (ndarray): Temporal derivative.
        Ix (ndarray): Spatial derivative along x.
        Iy (ndarray): Spatial derivative along y.
    """
    if deriv_filter is None:
        deriv_filter = np.array([1, -8, 0, 8, -1]) / 12  # Used in Wedel et al.

    # generate query indices
    H, W = img1.shape[:2]
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x2 = x + uv_prev[..., 0]
    y2 = y + uv_prev[..., 1]

    # do the same for matlab version --> different
    if eng is not None:
        x_matlab, y_matlab = eng.meshgrid(np.arange(W) + 1, np.arange(H) + 1, nargout=2)
        x2_matlab = x_matlab + uv_prev[..., 0]
        y2_matlab = y_matlab + uv_prev[..., 1]


    if interpolation_method in ['bi-linear', 'linear', 'cubic']: #todo: this is confusing, because linear ends up as cubic
        method = 'linear' if (interpolation_method == 'bi-linear' or interpolation_method == 'linear') else 'cubic'
        if len(img1.shape) == 2:  # Grayscale
            warpIm = interp2(img2, x2, y2, method=method,
                             method_selection=method_selection)  # throws errors here for multichannel
            if eng is not None:
                warpIm_matlab = eng.interp2(img2, x2_matlab, y2_matlab, method)
                assert np.allclose(warpIm, warpIm_matlab, equal_nan=True)
            It = warpIm - img1
            I2x = convolve(img2, deriv_filter[None, ::-1], mode='reflect')
            I2y = convolve(img2, deriv_filter[::-1, None], mode='reflect')
            if eng is not None:
                I2x_matlab = eng.imfilter(img2, deriv_filter[None,:],  'corr', 'symmetric', 'same')
                I2y_matlab = eng.imfilter(img2, deriv_filter[:,None],  'corr', 'symmetric', 'same')
                assert np.allclose(I2x_matlab, I2x)
                assert np.allclose(I2y_matlab, I2y)
            Ix = interp2(I2x, x2, y2, method=method, method_selection=method_selection)
            Iy = interp2(I2y, x2, y2, method=method, method_selection=method_selection)
            if eng is not None:
                Ix_matlab = eng.interp2(I2x, x2_matlab, y2_matlab, method)
                Iy_matlab = eng.interp2(I2y, x2_matlab, y2_matlab, method)
                assert np.allclose(Ix_matlab, Ix, equal_nan=True)
                assert np.allclose(Iy_matlab, Iy, equal_nan=True)
            I1x = convolve(img1, deriv_filter[None, ::-1], mode='reflect')
            I1y = convolve(img1, deriv_filter[::-1, None], mode='reflect')
        else:  # Color
            warpIm = np.zeros_like(img1)
            Ix = np.zeros_like(img1)
            Iy = np.zeros_like(img1)
            I1x = np.zeros_like(img1)
            I1y = np.zeros_like(img1)

            for j in range(img1.shape[2]):
                warpIm[..., j] = interp2(img2[..., j], x2, y2, method=method, method_selection=method_selection)
                I2x = convolve(img2[..., j], deriv_filter[None, ::-1], mode='reflect')
                I2y = convolve(img2[..., j], deriv_filter[::-1, None], mode='reflect')

                Ix[..., j] = interp2(I2x, x2, y2, method=method, method_selection=method_selection)
                Iy[..., j] = interp2(I2y, x2, y2, method=method, method_selection=method_selection)

                I1x[..., j] = convolve(img1[..., j], deriv_filter[None, ::-1], mode='reflect')
                I1y[..., j] = convolve(img1[..., j], deriv_filter[::-1, None], mode='reflect')

        if eng is not None:
            I1x_matlab = eng.imfilter(img1, deriv_filter[None, :], 'corr', 'symmetric', 'same')
            I1y_matlab = eng.imfilter(img1, deriv_filter[:, None], 'corr', 'symmetric', 'same')
            assert np.allclose(I1x_matlab, I1x, equal_nan=True)
            assert np.allclose(I1y_matlab, I1y, equal_nan=True)

        It = warpIm - img1

        # Temporal average
        Ix = b * Ix + (1 - b) * I1x
        Iy = b * Iy + (1 - b) * I1y

        # Handle out-of-boundary pixels
        mask = ~np.isfinite(It) | ~np.isfinite(Ix) | ~np.isfinite(Iy)
        It[mask] = 0
        Ix[mask] = 0
        Iy[mask] = 0

        return It, Ix, Iy

    else:
        raise ValueError('Unknown interpolation method!')