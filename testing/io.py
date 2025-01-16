import os
import pickle
import datetime
def get_eng():
    base = os.getenv("QBPY_BASE_DIR")
    if base is None:
        raise EnvironmentError("QBPY_BASE_DIR environment variable not set.")
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        # Add the folders and all subfolders recursively
        eng.addpath(eng.genpath(base), nargout=0)
        #eng.addpath(eng.genpath(base + "/qbp_matlab/burst"), nargout=0)
        #eng.addpath(eng.genpath(base + "/qbp_matlab/scripts"), nargout=0)
        #eng.addpath(eng.genpath(base + "/qbp_matlab/utils"), nargout=0)
        #eng.addpath(eng.genpath(base + "/qbp_matlab/single-photon-imaging"), nargout=0)
        #eng.addpath(eng.genpath(base + "/qbpy"), nargout=0)
    except ImportError:
        eng = None
        print("MATLAB engine not available. MATLAB tests will be skipped.")
    return eng


def save_inputs(func_name, inputs, with_timestamp=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if with_timestamp:
        filename = f"{func_name}_{timestamp}_inputs.pkl"
    else:
        filename = f"{func_name}_inputs.pkl"
    filepath = os.path.join(os.getenv("QBPY_BASE_DIR"),"testing", "test_data_inputs", filename)
    # skip, if the file exists
    if os.path.exists(filepath):
        return
    with open(filepath, "wb") as f:
        pickle.dump(inputs, f)
    print(f"Inputs saved to {filepath}")

def save_error(func_name, inputs, error_message, tag="error", with_timestamp=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if with_timestamp:
        filename = f"{func_name}_{timestamp}_{tag}_inputs.pkl"
    else:
        filename = f"{func_name}_error_inputs.pkl"
    filepath = os.path.join(os.getenv("QBPY_BASE_DIR"),"testing", "test_data_inputs", filename)
    with open(filepath, "wb") as f:
        pickle.dump({"inputs": inputs, "error": error_message}, f)
    print(f"Error logged to {filepath}")