function [initMatch, bestMatch] = dnc_upsampleMatches(bestMatch, tgtImg, finerPatchSize, coarserPatchSize, ratio, dataType)
% Upsample flow field from coarser level to finer level, initializing matches for next iteration.

hl = size(tgtImg,1) / finerPatchSize;
wl = size(tgtImg,2) / finerPatchSize;
assert(~mod(hl, 1) && ~mod(wl, 1));

initMatch = zeros([hl wl 3 2], dataType);

bestMatch = round(bestMatch * ratio);
yv = (0:hl-1)*finerPatchSize/ratio/coarserPatchSize;
xv = (0:wl-1)*finerPatchSize/ratio/coarserPatchSize;
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