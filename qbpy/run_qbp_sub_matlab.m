global json_path;

param = param_from_json(json_path);

[imbs, dcr, h5_info, dropped, phase_ids] = load_dataset(param);
results = qbpPipelineMono(imbs, param, dcr, dropped, phase_ids);
