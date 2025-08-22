#!/bin/sh
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load tools/prod GCCcore/14.3.0 CMake VTK

cd ../build
make -j16
