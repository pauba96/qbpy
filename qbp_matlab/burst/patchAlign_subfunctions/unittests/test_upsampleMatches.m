base = getenv('QBPY_BASE_DIR');
data = load(base+"/testing/test_data_inputs/upsample_matches_inputs.mat");
S = dnc_upsampleMatches(data.bestMatch, data.tgtImg, data.finerPatchSize, data.coarserPatchSize, data.ratio, data.dataType);