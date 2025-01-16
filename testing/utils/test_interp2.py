import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import os
import pickle
import unittest

from qbpy.burst.patchAlign_subfuns.dc_utils import save_to_mat
from testing.io import get_eng

from qbpy.utils.interp2 import interp2

class TestInterp2Functions(unittest.TestCase):
    def setUp(self):
        self.base = os.getenv("QBPY_BASE_DIR")
        if self.base is None:
            raise EnvironmentError("QBPY_BASE_DIR not set.")
        try:
            import matlab.engine
            self.eng = get_eng()
        except ImportError:
            self.eng = None
            print("MATLAB engine not available, tests involving MATLAB will be skipped.")
    def run_test(self, file_id, method="load", method_selection="load"):
        if self.eng is None:
            self.skipTest("MATLAB engine not available")
        # Load test data
        data_path = os.path.join(self.base, f"testing/test_data_inputs/interp2_{file_id}.pkl")
        if not os.path.isfile(data_path):
            self.skipTest("data not available")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # python version of patchAlignBinary
        if method == "load":
            method = data["method"]
        if method_selection == "load":
            method_selection = data["method_selection"]
        Vq_py = interp2(data['V'], data['Xq'], data['Yq'],method=method, method_selection=method_selection, indexing="mat")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        Vq_mat = self.eng.interp2(data['V'], data['Xq'], data['Yq'],method)

        self.plot_diff(data['V'], Vq_mat, Vq_py, np.abs(Vq_py - Vq_mat)>0, method, method, method_selection, file_id)

        # Compare flows_mat and flows_py
        try:
            np.testing.assert_allclose(Vq_py, Vq_mat, equal_nan=True)
        except Exception as e:
            #save as pkl and mat
            filepath = os.getenv("QBPY_BASE_DIR") + "/testing/test_data_inputs/interp2_assertion.pkl"
            save_to_mat(self.eng, filepath.replace(".pkl", ".mat"), V=data["V"], Xq=data["Xq"], Yq=data["Yq"], method=method,
                        method_selection=data['method_selection'], indexing=data['indexing'])
            with open(filepath, 'wb') as f:
                pickle.dump({'V': data["V"], 'Xq': data["Xq"], 'Yq': data["Yq"], 'method': method, 'method_selection': data["method_selection"],
                             'indexing': data['indexing']}, f)
            raise e

    def test_patch_merge_standard(self):
        self.run_test("inputs", "linear")

    def test_patch_merge_standard_nearest(self):
        self.run_test("inputs","nearest")

    def test_patch_merge_error(self):
        self.run_test("error")

    def test_patch_merge_assertion(self):
        self.run_test("assertion")

    def test_legacy(self):
        method = 'linear'
        method_selection = 'regular_grid'  # Choose between 'regular_grid', 'map_coordinates', 'rect_bivariate_spline', 'bicubic_convolution'
        if method != 'cubic':
            method_selection = 'regular_grid'
        eng = matlab.engine.start_matlab()

        # Create grid and values for interpolation
        V = np.random.rand(10, 15)

        # Query points for interpolation
        Xq = np.linspace(0, 9, 100)
        Yq = np.linspace(0, 9, 100)
        Xq_py, Yq_py = np.meshgrid(Xq, Yq, indexing='ij')
        Xq_matlab, Yq_matlab = np.meshgrid(Xq + 1, Yq + 1, indexing='xy')

        # Interpolate in Python
        Vq = interp2(V, Xq_py, Yq_py, method=method, method_selection=method_selection, indexing="py")

        # Compare with MATLAB
        matlab_method = 'linear'
        Vq_matlab = eng.interp2(V, Xq_matlab, Yq_matlab, matlab_method)
        Vq_matlab = np.array(Vq_matlab)
        self.plot_diff(V, Vq_matlab, Vq, np.abs(Vq - Vq_matlab), method, matlab_method, method_selection,
                       "legacy_py_indices")
        assert np.allclose(Vq, Vq_matlab, equal_nan=True)

        Vq_matlab_indices = interp2(V, Xq_matlab, Yq_matlab, method=method, method_selection=method_selection, indexing="mat")
        self.plot_diff(V, Vq_matlab_indices, Vq, np.abs(Vq - Vq_matlab_indices), method, matlab_method, method_selection, "legacy_mat_indices")
        assert np.allclose(Vq, Vq_matlab_indices, equal_nan=True)

        # Calculate the difference
        diff = np.abs(Vq - Vq_matlab)
        if np.sum(np.isnan(diff)) > 0:
            diff[np.isnan(diff)] = 1
        if np.sum(np.isnan(Vq_matlab)) > 0:
            Vq_matlab[np.isnan(Vq_matlab)] = 1



        assert np.allclose(Vq, Vq_matlab, equal_nan=True)


    def plot_diff(self,V,Vq_matlab,Vq,diff,method,matlab_method,method_selection,name):
        # copy all inputs to prevent them from changing
        V = V.copy()
        Vq = Vq.copy()
        Vq_matlab = np.array(Vq_matlab)
        Vq_matlab = Vq_matlab.copy()

        # replace Nans with 0
        Vq[np.isnan(Vq)] = 0
        Vq_matlab[np.isnan(Vq_matlab)] = 0
        diff = np.abs(Vq_matlab-Vq)



        plt.figure()
        plt.subplot(221)
        plt.imshow(Vq_matlab)
        plt.title("matlab")
        plt.subplot(222)
        plt.imshow(Vq)
        plt.title("python")
        plt.subplot(223)
        plt.imshow(np.clip(diff, -0.01, 0.5), cmap='hot')
        plt.title("difference")
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(V)
        plt.title("original")
        plt.suptitle(f"{method}-{matlab_method} interpolation - {method_selection}")
        plt.tight_layout()
        #save image
        basepath = os.path.join(os.path.dirname(__file__), "..", "results")
        plt.savefig(basepath + f"diff_{method}-{matlab_method}_{method_selection}_{name}.png")
        plt.close("all")
        print(f"{method}-{matlab_method} interpolation - {method_selection}")
