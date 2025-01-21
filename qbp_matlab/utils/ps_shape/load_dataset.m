function [imbs, dcr, h5_info, dropped, phase_ids] = load_dataset(param)
    h5_info = struct();  % Struct to store HDF5 file information
    dataDir = param.dataDir;  % Path to the data directory or file
    dcrPath = param.dcrPath;  % Path to the dark count rate file
    dataset_type = param.dataset_type;  % Type of dataset ("qbp", "npy", "h5")
    targetSize = param.target_size;  % Desired size of output images
    num_ls = param.num_ls;  % Number of light sources
    PS = param.PS;  % Flag for photometric stereo
    resultDir = param.resultDir;
    
    if dataset_type == "h5"
        % Load data from HDF5 file
        tic;
        h5_file_path = dataDir;  % Path to the HDF5 file
        disp(h5_file_path);
      
        % Check for valid dimensions (3 or 4) and permute the data accordingly
        data_path = '/photon_cube';  % Path to the dataset within the HDF5 file
        data = h5read(h5_file_path, data_path);  % Read the dataset
        
        % Reshape the data to match the desired target size and format
        % Determine the number of dimensions
        numDims = ndims(data);
        assert(numDims == 3);
        % Single light modality: 512x512x801 -> 801x512x512
        data = permute(data, [3, 2, 1]);
        t = size(data, 3); % Number of time slices
        imbs = cell(1, t);
        for i = 1:t
            % Extract and reshape each slice
            % currentSlice = logical(data(:, :, i)); %removed for AABB7
            currentSlice = data(:, :, i);
            currentSlice = squeeze(currentSlice);
            % Pad the current slice if necessary
            paddedSlice = padarray(currentSlice, [targetSize(1) - size(currentSlice, 1), targetSize(2) - size(currentSlice, 2)], 0, 'post');
            % Store the padded slice in the cell array
            imbs{i} = paddedSlice;
        end
        
        % Attempt to read the dark count rate if available
        try
            dcr_path = '/dcr';
            dcr = h5read(h5_file_path, dcr_path);  % Read the DCR if available
        catch
            dcr = zeros(targetSize(1), targetSize(2));  % Fallback if DCR is not present
            fprintf('Dark count rate not found in HDF5 file. Using default zeros.\n');
        end

        
        dropped_path = '/meta_dropped';  % Path to the dataset within the HDF5 file
        dropped = h5read(h5_file_path, dropped_path);  % Read the dataset
        
        if PS
            phase_ids_path = '/meta_phase_ids';  % Path to the dataset within the HDF5 file
            phase_ids = h5read(h5_file_path, phase_ids_path);  % Read the dataset
        else
            phase_ids = false; %zeros(1,numel(imbs));
        end

        % Save relevant HDF5 information to the struct
        h5_info.file_path = h5_file_path;
        h5_info.data_path = data_path;
        
        toc;
        fprintf('Finished reading HDF5 data.\n');

    elseif dataset_type == "mat"              
        %% Read images
        tic;
        range = param.range;
        imbs = ss2_1b_range_read(dataDir, range(1), range(2));
        load(param.dcrPath, 'dcr');
        toc;
        fprintf('Finished reading images.\n');
        dropped = zeros(size(imbs));
        h5_info = false;
        phase_ids = false;
    else
        error('Unsupported dataset type: %s', dataset_type);
    end
end
