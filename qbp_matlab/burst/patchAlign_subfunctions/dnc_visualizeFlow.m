function dnc_visualizeFlow(bestMatch, size_next_level, resultDir, level)
H = size_next_level(1);
W = size_next_level(2);
flowwarp = repelem(bestMatch,H/size(bestMatch,1),W/size(bestMatch,2),1);
flowhsv = drawFlowHSV(flowwarp);
imwrite(flowhsv, fullfile(resultDir, sprintf('flow-temp-l%d.png', level)));
end