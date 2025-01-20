import h5py
import numpy as np

def load_dataset(param):
    h5_info = {}  # Dictionary to store HDF5 file information
    data_dir = param['dataDir']  # Path to the data directory or file
    dcr_path = param['dcrPath']  # Path to the dark count rate file
    dataset_type = param['dataset_type']  # Type of dataset ("qbp", "npy", "h5")
    target_size = param['target_size']  # Desired size of output images
    num_ls = param['num_ls']  # Number of light sources
    PS = param['PS']  # Flag for photometric stereo

    if dataset_type == "h5":
        # Load data from HDF5 file
        h5_file_path = data_dir  # Path to the HDF5 file
        print(h5_file_path)

        # Check for valid dimensions (3 or 4) and permute the data accordingly
        data_path = '/photon_cube'  # Path to the dataset within the HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            data = f[data_path][:]  # Read the dataset

            # Reshape the data to match the desired target size and format
            # Determine the number of dimensions
            num_dims = data.ndim
            assert num_dims == 3
            # Single light modality: shape for example 512x512x801 -> convert to 801x512x512
            data = np.transpose(data, (2, 0, 1))
            t = data.shape[0]  # Number of time slices
            imbs = [np.pad(data[i], ((0, target_size[0] - data.shape[1]), (0, target_size[1] - data.shape[2])), 'constant') for i in range(t)]

            # Attempt to read the dark count rate if available
            try:
                dcr = f['/dcr'][:]  # Read the DCR if available
            except KeyError:
                dcr = np.zeros(target_size)  # Fallback if DCR is not present
                print('Dark count rate not found in HDF5 file. Using default zeros.')

            dropped = f['/meta_dropped'][:]  # Read the dropped dataset

            if PS:
                phase_ids = f['/meta_phase_ids'][:]  # Read the phase_ids dataset
            else:
                phase_ids = False # np.zeros(len(imbs))

        # Save relevant HDF5 information to the dictionary
        h5_info['file_path'] = h5_file_path
        h5_info['data_path'] = data_path

        print('Finished reading HDF5 data.')
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')

    return imbs, dcr, h5_info, dropped, phase_ids