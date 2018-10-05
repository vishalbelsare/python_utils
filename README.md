## Installation
The package is currently not available directly via PyPI. Instead, you can use: 

`pip install git+https://github.com/mloning/python_utils.git`

Alternatively, you can manually
1. Download the repository: `git clone https://github.com/mloning/python_utils.git`
2. Move into the repositor's root folder: `cd python_utils`
3. Install as pip package: `pip install .`

## Notes
A simpler way to make function scripts available in other Python files or 
notebooks (without creating a distributable Python package) is to simply add
 a `.pth` file containing the path to the folder of the scripts to the 
 Python  paths. This can be done by adding the `.pth` file to the 
 environment `site-package` directory, e.g. `~/
 .conda/envs/my_env/lib/python3.6/site-packages/`. Also see e.g. this 
 [StackOverflow questions](https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath/37008663#37008663)