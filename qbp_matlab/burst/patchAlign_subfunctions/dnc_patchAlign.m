function [flows] = dnc_patchAlign(ims, param, imv)
%PATCHALIGN Align intensity images using patch matching and Lucas-Kanade refinement

%Input:
%  ims: 1D cell array of images, (float, normalized to [0, 1])
%  param: struct that contains following fields:
%    refImage: index of the reference image in ims
%    numLevels: number of pyramid levels
%    patchSizes: array that contains patch sizes for each level
%    upsampleRatios: array that contains the upsample ratios for each pyramid level
%    searchRadii: array that contains the search radius at each level
%    numLKIters: number of Lucas-Kanade iterations for subpixel refinement
%    resultDir: directory to save results in
%    debug: whether or not to print debug information
%  imv: images used for debugging (warped imv will be saved)
%
%Output:
%  flows: 1D cell array of computed flows

%% Parameters and Setup
[H, W, C] = size(ims{1});
N = numel(ims);
refImage = param.refImage;

if nargin < 3 || isempty(imv)
    imv = ims;
end

if C == 1
    img = ims;
else
    % Convert to grayscale if not using ground-truth normals or multiple channels
    img = cell(1, N);
    for i = 1:N
        img{i} = rgb2gray(ims{i});
    end
end

resultDir = param.resultDir;
numLevels = param.numLevels;
patchSizes = param.patchSizes;
patchStride = patchSizes(1) / 2;
upsampleRatios = param.upsampleRatios;
searchRadii = param.searchRadii;
numStrides = patchSizes(1) / patchStride;
assert(~mod(numStrides, 1));
hs = (H - patchSizes(1)) / patchStride + 1;
ws = (W - patchSizes(1)) / patchStride + 1;
assert(~mod(hs, 1) && ~mod(ws, 1));
flows = cell(1, N);

% Build pyramid for reference image
P0 = buildAggrePyramid(img{refImage}, upsampleRatios);

%% Main Loop
for i = 1:N
    if i == refImage
        % Reference image has zero flow
        flows{i} = zeros([hs ws 2]);
        continue
    end
    if param.debug
        fprintf('Block %d: ', i);
    end
    timeBlockStart = tic;

    % Build pyramid for target image
    P1 = buildAggrePyramid(img{i}, upsampleRatios);

    % Perform coarse-to-fine alignment on pyramids
    bestMatch = dnc_coarseToFineMatch(P0, P1, patchSizes, searchRadii, upsampleRatios, param);

    % Refine at the finest level
    flows{i} = dnc_refineFinestLevel(P0{1}, P1{1}, bestMatch, patchSizes, patchStride, searchRadii(1), param);

    % Debug visualization (optional)
    if param.debug
        dnc_debugVisualization(flows{i}, imv{i}, H, W, param, resultDir, i, timeBlockStart, patchSizes(1), numStrides);
    end
end

end