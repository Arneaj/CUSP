#!/bin/sh
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=3:mem=8gb
#PBS -j oe
cd $PBS_O_WORKDIR

module load miniforge/3 > /dev/null 2>&1
module load tools/prod > /dev/null 2>&1
module load SciPy-bundle/2023.07-gfbf-2023a > /dev/null 2>&1
module load VTK > /dev/null 2>&1


echo

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
	"27000"
	"23100"
	"22500"
	"21900"
	"21000"
	"20100"
)

cd build
cmake .. > /dev/null 2>&1
make > /dev/null 2>&1
cd ..


for run_nb in "${runs[@]}"; do
	for t in "${timesteps[@]}"; do
    	echo Run"$run_nb"_"$t"

		start_time=$SECONDS
		mkdir data/Run"$run_nb"_"$t" > /dev/null 2>&1

		cd build
		./full_process /rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run"$run_nb" "$t"

		cd ../python
		python fit_to_analytical.py ../data/Run"$run_nb"_"$t"

		cd ..
		echo "Finished fitting the data point at Run$((run_nb)), timestep $((t))"
		echo "Time taken: $((SECONDS-start_time)) seconds total."

		echo
	done
done

# while read my_name; do
# 	my_path=../data/"$my_name"
# 	start_time=$SECONDS

# 	echo "Path to data: ${my_path}"

# 	cd ./output_cpp
# 	make full_process_test > /dev/null
# 	./full_process_test $my_path
# 	# make interest_points_test > /dev/null
# 	# ./interest_points_test $my_path
# 	cd ../python
# 	python fit_to_analytical.py $my_path
# 	cd ..
# 	echo "Finished fitting the data point at ${my_path}"
# 	echo "Time taken: $((SECONDS-start_time)) seconds total."

# 	echo
# done <<< "$($command)"

