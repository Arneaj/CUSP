#!/bin/sh

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
	"20100"
	"21000"
	"21900"
	"22500"
	"23100"
	"27000"
	"28800"
)

touch result_print_csv.csv
cat data/Run1_23100/analysis.csv | sed -n 1p > result_print_csv.csv

for run_nb in "${runs[@]}"; do
	for t in "${timesteps[@]}"; do
    	cat data/Run"$run_nb"_"$t"/analysis.csv | sed -n 2p >> result_print_csv.csv
		echo >> result_print_csv.csv
	done
done




