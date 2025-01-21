function bestMatch = dnc_findBestMatches(refImg, tgtImg, patchSize, searchRadius, initMatch, param)
% For each patch at the current pyramid level, find the best match and refine using LK.

dataType = param.dataType;
hl = size(initMatch,1);
wl = size(initMatch,2);
bestMatch = zeros([hl wl 2], dataType);

for j = 1:hl
    for k = 1:wl
        ylb = 1+(j-1)*patchSize;
        xlb = 1+(k-1)*patchSize;

        bestScore = Inf;
        currBest = [0 0]';
        for m = 1:size(initMatch, 3)
            [currMatch, currScore] = blockMatch2d(refImg, tgtImg, [ylb xlb], patchSize, searchRadius, initMatch(j,k,m,:));

            if currScore < bestScore * 0.9999
                % numerical errors are an issue here
                if abs(currScore - bestScore) < 0.0001
                    disp("warning: scores are dangerously close in findBestMatches!")
                end
                currBest = currMatch;
                bestScore = currScore;
            end
        end

        % LK refinement
        tempf = lkAlign(refImg(ylb:ylb+patchSize-1, xlb:xlb+patchSize-1), tgtImg,...
            param.numLKIters, cat(3,currBest(1),currBest(2))+cat(3,xlb-1,ylb-1));
        bestMatch(j,k,:) = tempf - [xlb-1 ylb-1];
    end
end

end