#!/bin/sh
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=32gb:ompthreads=16
#PBS -j oe
#PBS -N small_test
cd $PBS_O_WORKDIR

# echo $OMP_NUM_THREADS

module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1

# for reading the .pvtr files
module load VTK > /dev/null 2>&1		

# dependency of the Ceres least squares solver
module load Eigen > /dev/null 2>&1    	
module load Abseil > /dev/null 2>&1   				
module load googletest > /dev/null 2>&1   	


echo

cd build
# cmake .. > /dev/null
# make > /dev/null

valgrind ./full_process -i test_data \
						-t 23100 \
						--save_J_norm false \
						-X false -Y false -Z false \
						--save_ip false --save_params false





