function [bestMatch, bestScore] = blockMatch2d_multichannel(im0, im1, blockUL, blockSize, searchRadius, initMatch)
%BLOCKMATCH2D Brute-force block match in 2D for multi-channel images

if nargin < 6
    initMatch = zeros(1, 2);
end

[h, w, c] = size(im0);  % Add c for the number of channels

% Upper left corner coordinates of the block in im0
ylb = blockUL(1);
xlb = blockUL(2);

% Extract the reference patch from im0 (multi-channel)
refPatch = im0(ylb:ylb+blockSize-1, xlb:xlb+blockSize-1, :);

bestScore = inf;
bestMatch = [nan nan];

% Loop over search region
for v = max(1 - ylb, initMatch(2) - searchRadius) : min(h - blockSize + 1 - ylb, initMatch(2) + searchRadius)
    for u = max(1 - xlb, initMatch(1) - searchRadius) : min(w - blockSize + 1 - xlb, initMatch(1) + searchRadius)
        % Extract current patch from im1 (multi-channel)
        curPatch = im1(ylb + v : ylb + v + blockSize - 1, xlb + u : xlb + u + blockSize - 1, :);
        
        % Calculate the score as the sum of absolute differences across all channels
        curScore = sum(abs(curPatch - refPatch), 'all');
        
        % Update the best match if the current score is better
        if curScore < bestScore
            bestScore = curScore;
            bestMatch = [u v];
        end
    end
end

end
