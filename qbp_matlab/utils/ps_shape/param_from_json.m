function [param] = param_from_json(jsonFilePath)% Define default parameters
dataDir = "not set, provide in json";
dcrPath = "not set, provide in json";
resultDir = "not set, provide in json";
defaultParam = struct(...
    'dataDir',dataDir,...
    ...% Parameters to define the overall mode, we are running the data
    'dataset_type','qbp',... %how is the data structured, either qbp, npy, exr_folder*, h5py*
    'target_size',[512 512],...
    ...% These parameters determine which frames are used in align and merge
    'alignTWSize', 100, 'alignTWNum', 20,...
    'mergeTWSize', 100, 'mergeTWNum', 20, 'warpTWSize', 10,...
    'srTWSize', 40, 'srTWNum', 50,...
    'refFrame', calcRefFrame(100, 20),...
    ...% Parameters for align and merging, don't change for now
    'numLevels', 3, 'patchSizes', [16 16 8],...
    'upsampleRatios', [1 2 4], 'searchRadii', [1 4 8], 'numLKIters', 3,...
    'imgScale', 1, 'imgAutoScale', true,...
    'wienerC', 8,...
    'flowLambda', 0.01,...
    ...% Parameters for super-resolution, don't change for now
    'srScale', 2, 'combineRadius', 1,...
    'k_detail', 0.3, 'k_denoise', 1, 'D_th', 0.005, 'D_tr', 0.5, 'k_stretch', 1, 'k_shrink', 1,...
    'wienerSRC', 8,...
    ...% Parameters for post-denoising
    'bm3dSigma', 0,...
    ...% Parameters for correcting hot pixels, not used for simulation
    'hpThresh', 50, 'correctDCR', false, 'removeHP', true,...
    'dcrPath', dcrPath,...
    ...% Configuration, keep doRefine and deRefineSR false for now
    'fastMode', true, 'dataType', 'double',...
    'doRefine', false, 'doSR', false, 'doRefineSR', false,...
    'computePSNR', false,...
    'debug', false, 'saveImages', true, 'resultDir', resultDir,...
    ...% Photometric Stereo parameters
    'PS', false, 'num_ls', 1, 'image_path', dataDir, 'use_gt_flow',...
    "not defined", 'calculate_flow',true,'debug_flow',true,'perFrameGt',false, ...
    'loop_lvl',0);

% Load parameters from JSON file if provided
if ~isempty(jsonFilePath)
    if isfile(jsonFilePath)
        % Read JSON file
        jsonData = jsondecode(fileread(jsonFilePath));
        
        % Check if the "qbp" field is present
        if isfield(jsonData, 'qbp')
            qbpData = jsonData.qbp;
            % Merge all fields from qbpData into defaultParam
            jsonFields = fieldnames(qbpData);
            for k = 1:length(jsonFields)
                fieldValue = qbpData.(jsonFields{k});
                
                % If the field contains a string, check for environment variables
                if ischar(fieldValue)
                    % Replace any environment variables in the field value
                    fieldValue = envsubst(fieldValue);
                elseif iscell(fieldValue)  % If its a cell array, check each element
                    for i = 1:numel(fieldValue)
                        if ischar(fieldValue{i})
                            fieldValue{i} = envsubst(fieldValue{i});
                        end
                    end
                end
                
                % Assign the (potentially modified) value back to defaultParam
                defaultParam.(jsonFields{k}) = fieldValue;
            end
        else
            error('Specified JSON file does not contain "qbp" field.');
        end
        if isfield(jsonData, 'dataset')
            defaultParam.n_binary = jsonData.dataset.n_binary;
        else
            defaultParam.n_binary = 1;
        end
    else
        error('Specified JSON file does not exist.');
    end
end


% Use the merged parameters
param = defaultParam;
end