function bestMatch = dnc_coarseToFineMatch(P0, P1, patchSizes, searchRadii, upsampleRatios, param)
% Perform coarse-to-fine block matching and LK refinement at each pyramid level.

numLevels = param.numLevels;
dataType = param.dataType;

% Initialize at coarsest level
hl = size(P1{numLevels},1) / patchSizes(numLevels);
wl = size(P1{numLevels},2) / patchSizes(numLevels);
assert(~mod(hl, 1) && ~mod(wl, 1));
initMatch = zeros([hl wl 1 2], dataType);

for l = numLevels:-1:2
    if param.debug
        fprintf('L%d:', l);
    end

    % Find best matches at level l
    bestMatch = dnc_findBestMatches(P0{l}, P1{l}, patchSizes(l), searchRadii(l), initMatch, param);

    % Upsample results for next (finer) level, if not at the second level yet
    if l > 2
        [initMatch, bestMatch] = dnc_upsampleMatches(bestMatch, P1{l-1}, patchSizes(l-1), patchSizes(l), upsampleRatios(l), dataType);
    end

    % Debug (optional)
    if param.debug
        dnc_visualizeFlow(bestMatch, size(P1{l-1}), param.resultDir, l);
    end
end

% Return the best match for the second finest level
bestMatch = bestMatch;
end