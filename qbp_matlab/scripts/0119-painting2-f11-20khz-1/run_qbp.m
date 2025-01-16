%% Set path
filePath = mfilename('fullpath');
[sourceDir, scriptFname, ~] = fileparts(filePath);
[~, projName, ~] = fileparts(sourceDir);

dataDir = fullfile('/p/QuantaBurstPhotography/V1', 'siggraph20', projName, 'binary_images');
resultDir = pwd;

dcrPath = '/p/QuantaBurstPhotography/V1/Real/0111-dark-100khz-1/results0111/dcr.mat';

%% Quanta Parameters
param = struct(...
    ...% These parameters determine which frames are used in align and merge
    'alignTWSize', 500, 'alignTWNum', 40,...
    'mergeTWSize', 500, 'mergeTWNum', 40, 'warpTWSize', 10,...
    'srTWSize', 200, 'srTWNum', 100,...
    'refFrame', calcRefFrame(500, 40),...
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
    'hpThresh', 100, 'correctDCR', false, 'removeHP', true,...
    'dcrPath', dcrPath,...
    ...% Configuration, keep doRefine and deRefineSR false for now
    'fastMode', false, 'dataType', 'double',...
    'doRefine', false, 'doSR', true, 'doRefineSR', false,...
    'computePSNR', false,...
    'debug', false, 'saveImages', true, 'resultDir', resultDir);

%% Read images
tic;
imbs = ss2_1b_range_read(dataDir, 83750, 103749);
load(param.dcrPath, 'dcr');
toc;
fprintf('Finished reading images.\n');

%% Run pipeline
result = qbpPipelineMono(imbs, param, dcr);
