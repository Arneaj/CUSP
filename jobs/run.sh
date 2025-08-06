#!/bin/sh
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=16:mem=32gb:ompthreads=16
#PBS -j oe
#PBS -N mp_topology
cd $PBS_O_WORKDIR

exec > ../.result_folder/${PBS_JOBNAME}.o${PBS_JOBID%%.*} 2>&1
rm ${PBS_JOBNAME}.o${PBS_JOBID%%.*} 2>&1

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

echo Finished loading modules.

echo

runs=(
	"1"
	"2"
	"3"
	"4"
	"5"
	"6"
	"7"
	"8"
	"9"
	"10"
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

# cd build
# cmake .. > /dev/null
# make > /dev/null
# cd ..


# echo Finished building program.


for run_nb in "${runs[@]}"; do
	for t in "${timesteps[@]}"; do
    	echo Run"$run_nb"_"$t"

		start_time=$SECONDS
		mkdir data/Run"$run_nb"_"$t" > /dev/null 2>&1

		cd build
		./full_process 	-i /rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run"$run_nb"/MS \
						-s /rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run"$run_nb"/MS_Vars.csv \
						-t "$t" \
						-o /rds/general/user/avr24/home/Thesis/data/Run"$run_nb"_"$t" \
						--timing true \
						--logging true \
						--warnings true \
						--save_J_norm true \
						--save_Rho true

		cd ..
		# echo "Finished fitting the data point at Run$((run_nb)), timestep $((t))"
		# echo "Time taken: $((SECONDS-start_time)) seconds total."

		echo
	done
done



