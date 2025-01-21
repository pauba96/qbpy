base = getenv('QBPY_BASE_DIR');
data = load(base+"/testing/test_data_inputs/block_match_2d_inputs.mat");
flows_dnc = blockMatch2d(data.im0, data.im1, data.blockUL, data.blockSize, data.searchRadius, data.initMatch);