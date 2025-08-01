#!/bin/sh
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=32gb:ompthreads=16
#PBS -j oe
#PBS -N result_read_and_write
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

./read_and_write 	/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run10/MS/x00_rho-28800.pvtr \
					/rds/general/user/avr24/home/Thesis/test_data/rho_10_28800.bin





