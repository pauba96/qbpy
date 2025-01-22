import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import map_coordinates
from testing.TestFunctions import test_logger


@test_logger
def interp2(V, Xq, Yq, method='cubic', method_selection='regular_grid', indexing="py", X=None, Y=None):
    """
    INTERP2 2-D interpolation for uniformly-spaced data.
    Vq = interp2(X, Y, V, Xq, Yq, method='cubic', method_selection='regular_grid') interpolates
    to find Vq, the values of the underlying 2-D function V at the query points in matrices
    Xq and Yq, using bicubic interpolation.

    Parameters:
    - X, Y: Coordinates of the original grid.
    - V: Values at the grid points.
    - Xq, Yq: Coordinates of the query points.
    - method: Interpolation method, default is 'cubic'.
    - method_selection: Choose the interpolation method.
        - 'regular_grid' (default) for RegularGridInterpolator.
        - 'map_coordinates' for ndimage.map_coordinates.
        - 'rect_bivariate_spline' for RectBivariateSpline.
        - 'bicubic_convolution_*' for custom bicubic convolution interpolation.
          * denotes padding selection. Options are: 'reflect', 'constant', 'wrap', 'nearest'.
    """
    # copy inputs to prevent them from changing
    V = V.copy()
    Xq = Xq.copy()
    Yq = Yq.copy()


    if indexing == "mat": # if top-level calls the function with matlab indexing
        Xq -=1
        Yq-=1
        # switch Xq and Yq
        assert X is None, "currently not implemented"

    if (X is None) or (Y is None):
        assert (X is None) and (Y is None)
        # Create grid and values for interpolation
        X = np.arange(0, V.shape[0])
        Y = np.arange(0, V.shape[1])
        X, Y = np.meshgrid(X, Y, indexing='ij')

    # Ensure no NaNs in V
    assert np.sum(np.isnan(V)) == 0

    if method == 'nearest':
        # Using scipy.ndimage.map_coordinates
        points = np.array([Yq.ravel(), Xq.ravel()])
        Vq = map_coordinates(V, points, order=0, mode='constant',cval=np.nan).reshape(Xq.shape)

    elif method_selection == 'rect_bivariate_spline':
        # Using scipy.interpolate.RectBivariateSpline
        if X.ndim == 2 and Y.ndim == 2:
            X = X[:, 0]
            Y = Y[0, :]
        spline = RectBivariateSpline(X, Y, V, kx=3, ky=3)
        # Switch Xq and Yq
        Xq_, Yq_ = Xq, Yq
        Xq, Yq = Yq_, Xq_
        # Interpolate
        Vq = spline(Xq, Yq, grid=False)

    elif 'bicubic_convolution' in method_selection:
        padding_mode = method_selection.split("_")[-1]
        # Bicubic convolution interpolation using a custom kernel
        Vq = bicubic_convolution_interpolation(V, Yq, Xq, padding_mode=padding_mode)

    else:
        # Using scipy.interpolate.RegularGridInterpolator
        if X.ndim == 2 and Y.ndim == 2:
            X = X[:, 0]
            Y = Y[0, :]

        # Construct the interpolant
        F = RegularGridInterpolator((X, Y), V, method=method, bounds_error=False, fill_value=np.nan)

        # Switch Xq and Yq
        Xq_, Yq_ = Xq, Yq
        Xq, Yq = Yq_, Xq_

        # Now interpolate
        iscompact = compactgridformat(Xq, Yq)
        if iscompact:
            Xq = Xq[0, :]
            Yq = Yq[:, 0]
            Xq, Yq = np.meshgrid(Xq, Yq, indexing='ij') # xy to match matlab convention
            Vq = F(np.array([Xq.ravel(), Yq.ravel()]).T).reshape(Xq.shape)
        else:
            points = np.array([Xq.ravel(), Yq.ravel()]).T
            Vq = F(points).reshape(Xq.shape)

    return Vq

def bicubic_convolution_interpolation(V, Xq, Yq, padding_mode='reflect', constant_value=0):
    """
    Custom bicubic convolution interpolation with different padding options.

    Parameters:
    - V: The input 2D array.
    - Xq, Yq: Query points for interpolation.
    - padding_mode: Type of padding to handle edges. Options are:
        - 'constant': Pad with a constant value (specified by constant_value).
        - 'reflect': Reflect the values at the boundary.
        - 'wrap': Wrap around to the other side.
        - 'nearest': Repeat the edge values.
    - constant_value: The value to use for 'constant' padding mode.

    Returns:
    - Vq: The interpolated values at the query points.
    """
    # Define the cubic convolution kernel
    def cubic_kernel(x):
        abs_x = np.abs(x)
        abs_x2 = abs_x ** 2
        abs_x3 = abs_x ** 3

        return np.where(
            abs_x <= 1,
            (1.5 * abs_x3 - 2.5 * abs_x2 + 1),
            np.where(
                abs_x < 2,
                (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2),
                0
            )
        )

    # Pad the original data based on the chosen padding mode
    if padding_mode == 'constant':
        V_padded = np.pad(V, 2, mode='constant', constant_values=constant_value)
    elif padding_mode == 'reflect':
        V_padded = np.pad(V, 2, mode='reflect')
    elif padding_mode == 'wrap':
        V_padded = np.pad(V, 2, mode='wrap')
    elif padding_mode == 'nearest':
        V_padded = np.pad(V, 2, mode='edge')
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}. Choose from 'constant', 'reflect', 'wrap', 'nearest'.")

    # Prepare output array
    Vq = np.zeros(Xq.shape)

    # Perform bicubic convolution interpolation
    for i in range(Vq.shape[0]):
        for j in range(Vq.shape[1]):
            x = Xq[i, j]
            y = Yq[i, j]

            # Determine the four nearest neighbors in each direction
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            # Accumulate weighted contribution from each neighbor
            for m in range(-1, 3):
                for n in range(-1, 3):
                    # Adjust indices to access padded data correctly
                    x_idx = x0 + m + 2  # +2 due to padding on both sides
                    y_idx = y0 + n + 2

                    # Compute the distance to the neighbor
                    dx = x - (x0 + m)
                    dy = y - (y0 + n)

                    # Add contribution from the neighboring point
                    Vq[i, j] += V_padded[y_idx, x_idx] * cubic_kernel(dx) * cubic_kernel(dy)

    return Vq


# Helper function to strip NaNs for interpolation
def stripnanwrapper(X, Y, V):
    inan = np.isnan(V)
    jnan = np.any(inan, axis=0)
    inan = np.any(inan, axis=1)
    ncolnan = np.sum(jnan)
    nrownan = np.sum(inan)
    if ncolnan == 0 and nrownan == 0:
        return X, Y, V

    # Minimize loss of data, strip rows instead of columns if there are fewer rows
    if ncolnan > nrownan:
        Y, X, V = stripnansforspline(Y, X, V.T)
        V = V.T
    else:
        X, Y, V = stripnansforspline(X, Y, V)

    print('Warning: NaN values stripped during interpolation.')
    if V.size == 0 or V.ndim == 1:
        raise ValueError('Not enough points after NaN strip.')

    return X, Y, V


# Helper function to determine if the grid format is compact
def compactgridformat(X, Y):
    return (X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape and
            np.all(np.diff(X, axis=0) == 0) and np.all(np.diff(Y, axis=1) == 0))


# Placeholder for stripnansforspline function
def stripnansforspline(X, Y, V):
    # This function should remove rows/columns containing NaNs to minimize data loss
    mask = ~np.isnan(V)
    valid_rows = np.any(mask, axis=1)
    valid_cols = np.any(mask, axis=0)
    return X[valid_rows,:], Y[:,valid_cols], V[np.ix_(valid_rows, valid_cols)]