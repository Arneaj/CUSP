module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1

# for reading the .pvtr files
module load VTK > /dev/null 2>&1		

# dependency of the Ceres least squares solver
module load Eigen > /dev/null 2>&1    	
module load Abseil > /dev/null 2>&1   				
module load googletest > /dev/null 2>&1  

# to create the library 
module load pybind11