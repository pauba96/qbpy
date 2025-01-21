base = getenv('QBPY_BASE_DIR');
data = load(base+"/testing/test_data_inputs/patch_align_inputs.mat");
flows_dnc = dnc_patchAlign(data.blockAggres, data.param, data.blockRecons);
% flows = patchAlign(data.blockAggres, data.param, data.blockRecons);