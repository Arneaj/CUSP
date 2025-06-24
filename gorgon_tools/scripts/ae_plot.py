"""Todo: Docstring for ae_plot."""
import datetime as dt
import sys

import numpy as np

from ..geomagnetic.benchmarks import plot_benchmarks
from ..geomagnetic.coordinates import SM_to_SMD
from ..ionosphere.IS_import import ionosphere
from ..magnetosphere.gorgon_import import gorgon_sim

# load simulation data
path_to_data = sys.argv[1]  # previously used array job for shock runs
sim = gorgon_sim(path_to_data)  # read in simulation data for specific index
iono = ionosphere(path_to_data)

# define simulation time
starttime = 7200  # Simulation time in seconds from which to produce plots...
endtime = sim.times[-1]  # ... until a chosen time (by default the whole run)
dt_str_format_in = "%Y-%m-%d_%H:%M:%S"  # Format of datettime read in by script
t0_UT = dt.datetime.strptime(sys.argv[2], dt_str_format_in) - dt.timedelta(
    hours=2
)  # UT time corresponding to zero simulation time
times_UT = [
    t0_UT + dt.timedelta(seconds=int(t)) for t in iono.times
]  # full array of UT times
_, _, _, mu, del_mu = SM_to_SMD(
    0, 0, 1, np.array(times_UT)
)  # recalculate simulation dipole tilt angle
iono.tilt = mu  # store for use by calc_deltaB function

plot_benchmarks(
    iono,
    sim,
    starttime,
    endtime,
    t0_UT,
    data_folder=path_to_data + "/Data/" + sys.argv[3] + "/",
    plot_CPCP=True,
    rolling=False,
    plot_FPC=True,
    wdc_data="WWW_aeasy01863067.dat",
    disp=False,
    filename="Gorgon_Benchmarks.png",
    fileformat="png",
)
