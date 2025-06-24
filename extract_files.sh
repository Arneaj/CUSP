#!/bin/sh
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load miniforge/3

module load tools/prod
# module load intel/2023a
# module load Python/3.11.5-GCCcore-13.2.0
module load SciPy-bundle/2023.07-gfbf-2023a
# source ~/venv/venv1/bin/activate
# pip install -r requirements-linux.txt
# pip install numpy --update
# pip install matplotlib --update



timesteps=(
	"28800"
	"21000"
	"28200"
	"27600"
)

for run_nb in $(seq 1 9); do
	for t in "${timesteps[@]}"; do
    		echo Run"$run_nb"_"$t"
		mkdir data/Run"$run_nb"_"$t"
		python import_data.py Run"$run_nb" "$t"
		echo
	done
done
