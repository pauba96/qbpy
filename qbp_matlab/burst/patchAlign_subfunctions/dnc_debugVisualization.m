function dnc_debugVisualization(flow, imv, H, W, param, resultDir, i, timeBlockStart, patchSize, numStrides)
% Debug visualization of flow and warped images.

[X, Y] = meshgrid(1:W, 1:H);
flowwarp = repelem(flow(1:numStrides:end,1:numStrides:end,:),patchSize,patchSize,1);
flowhsv = drawFlowHSV(flowwarp);
imwrite(flowhsv, fullfile(resultDir, sprintf('flow%d-l%d.png', i, 1)));

if ndims(imv) == 2
    imvWarped = interp2(imv, X+flowwarp(:,:,1), Y+flowwarp(:,:,2), 'cubic');
else
    imvWarped = zeros(H, W, size(imv,3), param.dataType);
    for c = 1:size(imv,3)
        imvWarped(:,:,c) = interp2(imv(:,:,c), X+flowwarp(:,:,1), Y+flowwarp(:,:,2), 'cubic');
    end
end
imwrite(imvWarped, fullfile(resultDir, sprintf('imWarped%d-l%d.png', i, 1)));
fprintf('%g\n', toc(timeBlockStart));
end
