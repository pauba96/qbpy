import numpy as np
def save_to_mat(matlab_engine, filename, **kwargs):
	"""
	Save variables from Python to a .mat file via MATLAB engine.

	Parameters:
	matlab_engine : matlab.engine.MatlabEngine
		Active MATLAB engine instance.
	filename : str
		Name of the .mat file to save.
	kwargs : dict
		Key-value pairs where the key is the variable name (str) and the value is the data (numpy array, list, etc.).

	Example:
	save_to_mat(self.eng, 'data.mat', refImg=refImg, tgtImg=tgtImg, patchSize=5, searchRadius=3)
	"""
	if matlab_engine is None:
		# warning
		print("MATLAB engine not available")
		return

	for var_name, value in kwargs.items():
		# Convert numpy arrays to MATLAB double arrays
		if isinstance(value, np.ndarray):
			value = matlab_engine.double(value)

		# Assign the variable to MATLAB workspace
		matlab_engine.workspace[var_name] = value

	# Generate the save command dynamically
	vars_to_save = ', '.join([f"'{key}'" for key in kwargs.keys()])
	save_command = f"save('{filename}', {vars_to_save});"

	# Execute the save command
	matlab_engine.eval(save_command, nargout=0)

	print(f"Saved to {filename}: {', '.join(kwargs.keys())}")