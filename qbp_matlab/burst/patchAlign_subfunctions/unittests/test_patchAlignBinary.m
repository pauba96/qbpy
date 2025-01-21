base = getenv('QBPY_BASE_DIR');
data = load(base+"/testing/test_data_inputs/patch_align_binary_inputs.mat");
[flows, flowrs] = patchAlignBinary(data.imbs, data.param);