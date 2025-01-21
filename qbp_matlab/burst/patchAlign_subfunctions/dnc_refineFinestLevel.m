function finalFlow = dnc_refineFinestLevel(refImg, tgtImg, bestMatch, patchSizes, patchStride, searchRadius, param)
% Refine the flow field at the finest level using LK.

dataType = param.dataType;
H = size(refImg,1);
W = size(refImg,2);
hs = (H - patchSizes(1)) / patchStride + 1;
ws = (W - patchSizes(1)) / patchStride + 1;

initMatch = dnc_initializeMatchesFromLevel2(bestMatch, patchSizes, patchStride, param, size(refImg)); %this is wrong
finalFlow = zeros([hs ws 2], dataType);

for j = 1:hs
    for k = 1:ws
        ylb = 1+(j-1)*patchStride;
        xlb = 1+(k-1)*patchStride;

        bestScore = Inf;
        currBest = [0 0];
        for m = 1:size(initMatch, 3)
            [currMatch, currScore] = blockMatch2d(refImg, tgtImg, [ylb xlb], patchSizes(1), searchRadius, initMatch(j,k,m,:));
            if currScore < bestScore*0.9999
                currBest = currMatch;
                bestScore = currScore;
            end
        end

        % LK refinement if not in fast mode without SR
        if ~(param.fastMode && ~param.doSR)
            tempf = lkAlign(refImg(ylb:ylb+patchSizes(1)-1,xlb:xlb+patchSizes(1)-1), tgtImg,...
                param.numLKIters, currBest+cat(3,xlb-1,ylb-1));
            finalFlow(j,k,:) = tempf - [xlb-1 ylb-1];
        else
            finalFlow(j,k,:) = currBest;
        end
    end
end
end