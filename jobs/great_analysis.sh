#!/bin/sh
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=16:mem=32gb:ompthreads=16
#PBS -j oe
#PBS -N great_analysis
cd $PBS_O_WORKDIR

module load tools/prod GCCcore/14.3.0 CMake VTK	

echo Finished loading modules.
echo

cd ../build

./great_analysis  ../.result_folder/great_analysis${PBS_JOBID%%.*}.csv
