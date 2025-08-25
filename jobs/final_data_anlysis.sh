#!/bin/sh
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=16:mem=32gb:ompthreads=16
#PBS -j oe
#PBS -N final_data_analysis
cd $PBS_O_WORKDIR

cd ../python
python final_data_analysis.py
