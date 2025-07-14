#!/bin/sh
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1


runs=(
	"1"
	"2"
	"4"
	"5"
	"9"
	"11"
	"12"
	"13"
	"14"
	"15"
	"16"
	"17"
	"18"
	"19"
)


timesteps=(
	"28800"
	"28200"
	"27600"
	"27000"
	"23100"
	"22500"
	"21900"
	"21000"
)

for run_nb in "${runs[@]}"; do
	for t in "${timesteps[@]}"; do
    	echo Run"$run_nb"_"$t"
		mkdir data/Run"$run_nb"_"$t" > /dev/null 2>&1
		python import_data.py Run"$run_nb" "$t"
		echo
	done
done
