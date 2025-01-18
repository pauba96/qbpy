## What is this and why?
...

## Goal: Full implementation and Torch integration
...

## Notes
- You can look at the changes made to qbp by looking at this commit: ...
- Namely, we adjusted the partial_deriv_patch.m function to perform linear interpolation by default, as there is no exact python equivalent to the cubic interpolation used in Matlab. If you want to change this behavior, set the environment variable MATLAB_USE_CUBIC_INTERP to True
- We further divided the patchALign function into subfunctions to make testing easier

## Getting Started
1. set environment variable: QBPY_BASE_DIR to the directory where the repository is cloned to.
2. `conda create --name qbpy python=3.11`  Make sure your python version matches the matlab python api requirements of your Matlab version. (3.11 for R2024a)
3. Add QBPY_BASE_DIR to the PYTHONPATH in the activate script of the conda environment: `conda activate qbpy`
4. Optionally: Install Matlab in conda environment: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html 
`cd "matlabroot\extern\engines\python"´
`python -m pip install matlabengine`
note: needs admin privileges
5. Optional: follow the instructions on how to initialize qbp in qbp_matlab/README.md


## For developers
If you want to convert additional capabilites from Matlab to Python, here are some hints on how to do it:
1. Add the new function you want to implement using the Matlab API to the qbp package (eng.fun). And a wrapper for the inputs
basically: 
```python
def my_new_function(input1, input2):
    # wrap the inputs
    input1 = wrap(input1)
    input2 = wrap(input2)
    # call the matlab function
    output = eng.my_new_function(input1, input2)
    # unwrap the output
    output = unwrap(output)
    return output
```
2. Integrate this function into your pipeline (for instance, change the testing/experiment/test_run_qbp.py file to incorporate the new function)
3. replace eng.my_new_function with your python implementation
```python
from testing.TestFunctions import test_logger
@test_logger
def my_new_function(input1, input2):
    # do whatever my_new_function is doing, but in Python
    output = [i+input2 for i in input1] # for instance
    return output
```
4. Run the pipeline with the os.environ["ENABLE_TEST_LOGGER"] = 1 - this should generate for all functions with the @test_logger decorator a .pkl file in testing/test_data_input. Note: If you re-run the function and the .pkl file already exists, it will *not* be overwritten.
5. If there are errors, the inputs will be saved to a _error.pkl file in the same directory. You can use this file to debug the function.
6. Write a unittest that compares the output of the function to its Matlab equivalent. You can use the provided tools and look at the existing tests.

## Acknowledgements

## License