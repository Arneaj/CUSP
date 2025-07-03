#!/bin/sh
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1

python fit_to_analytical.py ../data/Run1_28800
