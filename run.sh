#!/bin/sh
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=3:mem=8gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1

command="ls data"

echo

while read my_name; do
	my_path=../data/"$my_name"
	start_time=$SECONDS

	echo "Path to data: ${my_path}"

	# cd ./output_cpp
	# make full_process_test > /dev/null
	# ./full_process_test $my_path
	cd ../python
	python fit_to_analytical.py $my_path
	cd ..
	echo "Finished fitting the data point at ${my_path}"
	echo "Time taken: $((SECONDS-start_time)) seconds total."

	echo
done <<< "$($command)"

