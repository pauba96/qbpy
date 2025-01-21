base = getenv('QBPY_BASE_DIR');
data = load(base+"/testing/test_data_inputs/patch_merge_binary_synthetic_warp.mat");
S = patchMergeBinary(data.imbs, data.flows, data.param, data.phase_ids);