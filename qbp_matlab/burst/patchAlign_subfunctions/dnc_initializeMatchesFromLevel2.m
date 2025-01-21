function initMatch = dnc_initializeMatchesFromLevel2(bestMatch, patchSizes, patchStride, param, ref_size)
% Initialize matches at the finest level from level-2 matches.

dataType = param.dataType;
H = ref_size(1);
W = ref_size(2);

hs = (H - patchSizes(1)) / patchStride + 1;
ws = (W - patchSizes(1)) / patchStride + 1;
initMatch = zeros([hs ws 3 2], dataType);

if param.numLevels > 1
    bestMatch = round(bestMatch * param.upsampleRatios(2));
    yv = (0:patchStride:H - patchSizes(1))/param.upsampleRatios(2)/patchSizes(2);
    xv = (0:patchStride:W - patchSizes(1))/param.upsampleRatios(2)/patchSizes(2);
    yr = floor(yv) + 1;
    xr = floor(xv) + 1;

    initMatch(:,:,1,1) = bestMatch(yr,xr,1);
    initMatch(:,:,1,2) = bestMatch(yr,xr,2);

    yrn = yr - 1;
    masky = mod(yv,1) >= 1/2 & yr < size(bestMatch,1) | yr == 1;
    yrn(masky) = yr(masky) + 1;
    initMatch(:,:,2,1) = bestMatch(yrn,xr,1);
    initMatch(:,:,2,2) = bestMatch(yrn,xr,2);

    xrn = xr - 1;
    maskx = mod(xv,1) >= 1/2 & xr < size(bestMatch,2) | xr == 1;
    xrn(maskx) = xr(maskx) + 1;
    initMatch(:,:,3,1) = bestMatch(yr,xrn,1);
    initMatch(:,:,3,2) = bestMatch(yr,xrn,2);
end
end
