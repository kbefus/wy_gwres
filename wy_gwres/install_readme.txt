# Readme for how to install Python 3.7 and required packages to use wy_gwres package

1. Install Python 3.7. Recommended to use Anaconda x64 distribution for best functionality. 
	- With Anaconda Python installed, it is also recommended to make an environment for installing packages
		(In Anaconda Prompt with sufficient permissions: conda create --name wygw)
		
2. Install required packages. All required packages and versions are included in wygw_environment.yml. They can be installed into the conda environment made above by activating the environment (conda activate wygw | conda install --file wygw_environment.yml) or directly making an environment with the package list (conda env create --file wygw_environment.yml --name wygw)

3. Spyder IDE is a convenient way to interact with and run scripts, acting as both a text editor and console. (install with "conda install spyder" in Anaconda prompt with wygw environment active)