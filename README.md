# Preprocess Functions for Data Analysis in Python

## Prerequisites
* Anaconda, manages Python environment and dependencies

## Installation
The package is currently not available directly via PyPi. Instead, you can use: 

`pip install git+https://github.com/mloning/preprocess_functions.git`

Or alternatively, you can
1. Download project: `git clone https://github.com/mloning/preprocess_functions.git`
2. Move into root folder: `cd preprocess_functions`
3. Install as pip package: `pip install .`

## Additional Notes
Another way to make the functions available in other Python scripts or notebooks (without creating a distributable Python package) is to add a `.pth` file containing the path to the folder of the scripts to the Python paths by adding the `.pth` file to the environment `site-package` directory, e.g. `~/.conda/envs/my_env/lib/python3.6/site-packages/`. For more information on how to make functions available in python or jupyter, see e.g.
https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath/37008663#37008663
