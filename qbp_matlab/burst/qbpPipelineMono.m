function [result] = qbpPipelineMono(imbs, param, dcr, imgt, phase_ids)
%QBPPIPELINEMONO Entire QBP pipeline for monochrome binary images
% based on srcMono_0212

% check if phase_ids are given
if nargin < 5 || isempty(phase_ids)
    phase_ids = false;
end


if ~isfield(param, 'normalize_output') || isempty(param.normalize_output)
    param.normalize_output = false; % Set to false by default
end

result = struct();
resultDir = param.resultDir;

%% Naive reconstruction with simple averaging
ima = naiveRecons(imbs, param);
if param.imgAutoScale
    [ima, param.imgScale] = autoScaleIntensity(ima, 97);
end
asParam = param;
asParam.mergeTWNum = 1;
refBlock = floor((param.refFrame - 1) / param.mergeTWSize);
imas = naiveRecons(imbs(refBlock*param.mergeTWSize+1:(refBlock+1)*param.mergeTWSize), asParam);

if param.removeHP
    tic;
    for i = 1:numel(imbs)
        fprintf(string(i)+"/"+string(numel(imbs))+ " ");
        imbs{i} = removeHotPixels(imbs{i}, dcr, param.hpThresh); 
    end
    toc;
    fprintf('Finished hot pixel correction.\n');
    
    imaf = naiveRecons(imbs, param);
    imasf = naiveRecons(imbs(refBlock*param.mergeTWSize+1:(refBlock+1)*param.mergeTWSize), asParam);
else
    imaf = [];
    imasf = [];
end
fprintf('Finished naive reconstruction.\n');

%% Align
[flows, flowrs] = patchAlignBinary(imbs, param);
if param.debug
    save(fullfile(resultDir, 'patchAlign.mat'), 'flows', 'flowrs', 'param');
end
result.flows = flows;
result.flowrs = flowrs;
fprintf('Finished alignment.\n');

%% Merge
Sr = patchMergeBinary(imbs, flows, param, phase_ids);
paramNoBM3D = param;
paramNoBM3D.bm3dSigma = 0;
imr = postMerge(Sr, paramNoBM3D, false);

if param.imgAutoScale
    [imr, imgScaleCorrect] = autoScaleIntensity(imr, 97);
    param.imgScale = param.imgScale * imgScaleCorrect;
    ima = ima * imgScaleCorrect;
    imas = imas * imgScaleCorrect;
    imaf = imaf * imgScaleCorrect;
    imasf = imasf * imgScaleCorrect;
end

if param.bm3dSigma > 0
    imrbm = postMerge(Sr, param, false);
    if param.saveImages
        imwrite(lin2rgb(imrbm), fullfile(resultDir, 'bm3d_g22.png'));
    end
else
    imrbm = [];
end


if param.saveImages
    % Check if imr is single-channel or multi-channel and handle appropriately
    if size(imr, 3) == 1
        % If imr is single-channel (grayscale), convert to RGB by replicating the channel
        imr_rgb = repmat(lin2rgb(imr), [1, 1, 3]);
    else
        % If imr is multi-channel, use the first three channels
        imr_rgb = lin2rgb(imr(:, :, 1:3));
    end
    % Write the image
    imwrite(imr_rgb, fullfile(resultDir, 'patchMerge_g22.png'));

    % Check if ima is single-channel or multi-channel and handle appropriately
    if size(ima, 3) == 1
        % If ima is single-channel, convert to RGB by replicating the channel
        ima_rgb = repmat(lin2rgb(ima), [1, 1, 3]);
    else
        % If ima is multi-channel, use the first three channels
        ima_rgb = lin2rgb(ima(:, :, 1:3));
    end
    % Write the image
    imwrite(ima_rgb, fullfile(resultDir, 'averageRecons_g22.png'));
end
%     imwrite(lin2rgb(imas), fullfile(resultDir, 'averageReconsShort_g22.png'));
%     if ~isfield(param, 'removeHP') || param.removeHP
%         imwrite(lin2rgb(imaf), fullfile(resultDir, 'averageReconsHPFixed_g22.png'));
%         imwrite(lin2rgb(imasf), fullfile(resultDir, 'averageReconsShortHPFixed_g22.png'));
%     end
if param.debug
    save(fullfile(resultDir, 'patchMerge.mat'), 'param', 'Sr', 'imr', 'imrbm');
    save(fullfile(resultDir, 'naiveRecons.mat'), 'ima', 'imaf', 'imas', 'imasf');
end
result.ima = ima;
result.imas = imas;
result.imaf = imaf;
result.imasf = imasf;
result.Sr = Sr;
result.imr = imr;
result.imrbm = imrbm;
result.param = param;
fprintf('Finished merging.\n');

%% Refine flow and merge
if param.doRefine
    Srr = patchMergeBinary(imbs, flowrs, param);
    imrr = postMerge(Srr, param, false);
    if param.debug
        save(fullfile(resultDir, 'patchMerge_refinedFlow.mat'), 'imrr', 'Srr');
    end
    if param.saveImages
        imwrite(lin2rgb(imrr), fullfile(resultDir, 'patchMerge_refinedFlow_g22.png'));
    end
    result.Srr = Srr;
    result.imrr = imrr;
    fprintf('Finished flow refinement.\n');
end

%% Superresolution
if param.doSR
    Ssr = patchWienerSR(imbs, flows, param, Sr);
    imsr = postMerge(Ssr, param, true);
    if param.debug
        save(fullfile(resultDir, 'patchWienerSR.mat'), 'imsr', 'Ssr');
    end
    if param.saveImages
        imwrite(lin2rgb(imsr), fullfile(resultDir, 'patchWienerSR_g22.png'));
    end
    result.Ssr = Ssr;
    result.imsr = imsr;
    fprintf('Finished super-resolution.\n');
end

if param.doRefineSR
    Ssrr = patchWienerSR(imbs, flowrs, param, Sr);
    imsrr = postMerge(Ssrr, param, true);
    if param.debug
        save(fullfile(resultDir, 'patchWienerSR_refinedFlow.mat'), 'imsrr', 'Ssrr');
    end
    if param.saveImages
        imwrite(lin2rgb(imsrr), fullfile(resultDir, 'patchWienerSR_refinedFlow_g22.png'));
    end
    result.Ssrr = Ssrr;
    result.imsrr = imsrr;
    fprintf('Finished super-resolution with flow refinement.\n');
end

%% Compute PSNR
if param.computePSNR
    psnr = struct();
    [H, W, ~] = size(imr);
    imgtr = imresize(imgt, [H W]);
    imgtr(imgtr>1) = 1;
    imgtr(imgtr<0) = 0;
    psnr.quantaNaivePSNR = evalPSNR(ima, imgtr);
    psnr.quantaBurstPSNR = evalPSNR(imr, imgtr);
    
    if param.doRefine
        psnr.quantaBurstRefinedPSNR = evalPSNR(imrr, imgtr);
    end
    
    if param.doSR
        [Hsr, Wsr, ~] = size(imsr);
        imgtsr = imresize(imgt, [Hsr Wsr]);
        psnr.quantaBurstSrPSNR = evalPSNR(imsr, imgtsr);
    end
    
    if param.doRefineSR
        [Hsr, Wsr, ~] = size(imsr);
        imgtsr = imresize(imgt, [Hsr Wsr]);
        psnr.quantaBurstSrRefinedPSNR = evalPSNR(imsrr, imgtsr);
    end
    
    if param.debug
        savejson('', psnr, fullfile(resultDir, 'quanta.json'));
    end
    result.psnr = psnr;
end
end

