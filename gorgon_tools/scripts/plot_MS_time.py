"""Module to provide a script for plotting magnetospheric data at a chosen time."""

import datetime as dt
import sys
import warnings

import numpy as np

from ..geomagnetic.visualise import plot_slices, stretched_to_uniform
from ..magnetosphere.gorgon_import import gorgon_sim

warnings.filterwarnings("ignore")

path_to_data = sys.argv[1]  # Directory containing simulation data
sim = gorgon_sim(path_to_data)  # Magnetosphere class
time = sys.argv[2]  # chosen time in seconds to plot

try:
    dt_str_format_in = "%Y-%m-%d_%H:%M:%S"  # Format of datettime read in by script
    dt_str_format_out = (
        "%Y-%m-%d_%H_%M"  # Format of datettime used for output file names
    )
    t0_UT = dt.datetime.strptime(sys.argv[3], dt_str_format_in)

    time_UT = t0_UT + dt.timedelta(seconds=int(time))  # UT time at timestep
    time_str = time_UT.strftime(dt_str_format_out)
    r_IB = float(sys.argv[4])
    mu = float(sys.argv[5]) * np.pi / 180
except Exception as e:
    time_UT = None
    time_str = str(time)
    r_IB = float(sys.argv[3])
    mu = float(sys.argv[4]) * np.pi / 180
    print(e)

# Import magnetospheric data at timestep
sim.import_timestep(sim.timestep(int(time)), ["rho", "Ti", "vvec", "Bvec_c"])
sim.arr["P"] = sim.arr["rho"] / 1.67e-27 * sim.arr["Ti"] * 1.6022e-19
sim.arr["vx"] = sim.arr["vvec"][:, :, :, 0]
if np.min(sim.dx) != np.max(sim.dx):
    stretched_to_uniform(sim, ["P", "vx", "Bvec_c"])

# Plotting parameters
plt_list = [
    [
        "P",
        {
            "name": r"$P$",
            "unit": r"nPa",
            "norm": 1e-9,
            "log": True,
            "min": 1e-4,
            "max": 1e1,
            "n_levs": 30,
            "cmap": "jet",
        },
    ],
    [
        "vx",
        {
            "name": r"$v_x$",
            "unit": r"kms$^{-1}$",
            "norm": 1e3,
            "min": -600,
            "max": 600,
            "log": False,
            "n_levs": 30,
            "cmap": "RdBu_r",
        },
    ],
]

plt_coords = "sim"  # change to 'GSM' if you uncommented the above
# And plot each variable...
for plts in plt_list:
    plot_slices(
        sim,
        plts,
        plt_coords,
        r_IB,
        mu=mu,
        t_UT=time_UT,
        disp=False,
        plot_flines=True,
        filename=path_to_data + "/Plots/MS/" + plts[0] + "_" + time_str,
        fileformat="jpg",
    )
