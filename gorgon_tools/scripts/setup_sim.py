"""Module to set up the simulation."""

import datetime as dt
import sys

import numpy as np

from gorgon_tools.solar_wind import (
    gen_gorgon_SW,
    get_SW_data,
    plot_SW,
    read_EUHFORIA_data,
    read_OMNIWeb_OMNI_data,
    read_SW_input,
)

# Usage: Setting up variable solar wind data from predetermined file. Note dependence on
# SW_Input.dat as input file.

# Define full data timerange
dt_str_format = "%Y-%m-%d %H:%M:%S"
starttime = dt.datetime.strptime(sys.argv[3], dt_str_format) - dt.timedelta(hours=2)
endtime = dt.datetime.strptime(sys.argv[4], dt_str_format) + dt.timedelta(minutes=10)
totaltime = (
    dt.datetime.strptime(sys.argv[4], dt_str_format) - starttime
).total_seconds()

# Load data from file for chosen source
if sys.argv[1] == "OMNI":
    if len(sys.argv[2]) > 2:
        SW_data = read_OMNIWeb_OMNI_data(sys.argv[2])
    else:
        SW_data = get_SW_data(starttime, endtime, "OMNI")
elif sys.argv[1] == "EUHFORIA":
    SW_data = read_EUHFORIA_data(sys.argv[2])

# Transform into Gorgon coordinates and write input file
SW_data = SW_data[starttime.strftime(dt_str_format) : endtime.strftime(dt_str_format)]
SW_data, _, _, sim_params = gen_gorgon_SW(
    SW_data,
    simtime=0,
    use_SMD=True,
    coords="GSM",
    max_yz_inflow=(20, 20),
    write=True,
    smoothing=False,
)

plot_SW(SW_data, coords="GSM", source="OMNI", show_UT=True, disp=False)

# Plot and save figure of input data
SW_data = read_SW_input("SW_Input.dat", starttime, coords="sim")
plot_SW(
    SW_data,
    coords="sim",
    source="Simulation Input",
    show_UT=False,
    simtime=0,
    disp=False,
)

# Save simulation parameters for time range
config_arr = np.array(
    [key + " = " + str(sim_params[key]) for key in sim_params.keys()]
).astype(str)
config_arr = np.append(["time_stop = " + str(totaltime)], config_arr)
config_arr = np.append(
    ["UT_datetime = " + starttime.strftime("%Y-%m-%d_%H:%M:%S")], config_arr
)
np.savetxt("ctrl_tmp.txt", config_arr, fmt="%s")
