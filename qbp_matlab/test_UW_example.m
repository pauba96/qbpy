base = getenv('QBPY_BASE_DIR');

json_path = base + "/Configs/single_UW_example.json";

param = param_from_json(json_path);

[imbs, dcr, h5_info, dropped, phase_ids] = load_dataset(param);

results = qbpPipelineMono(imbs, param, dcr, dropped, phase_ids);