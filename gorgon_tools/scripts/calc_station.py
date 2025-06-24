"""Todo: Docstring for calc_station."""
import datetime as dt
import sys

import numpy as np
import pandas as pd

from ..geomagnetic.calcdeltaB import calc_B_vectors
from ..geomagnetic.coordinates import SM_to_SMD
from ..geomagnetic.elecproject import elecproject
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
iono.time = 7200
iono.times = iono.times[iono.timestep(7200) :: 10]

ae_stations = np.genfromtxt("stations.tsv", dtype=str)
i = np.argwhere(ae_stations[:, 3] == sys.argv[4])[0][0]
station = [
    sys.argv[4],
    float(ae_stations[i, 1]) * np.pi / 180,
    float(ae_stations[i, 2]) * np.pi / 180,
]
print(station)
# colat,lon = get_station_coords(sys.argv[4])
# station = [sys.argv[4],colat,lon]
if sys.argv[3] == "CDB":
    output = pd.read_csv(
        path_to_data + "/Data/CDB/output" + sys.argv[4] + ".csv",
        parse_dates=True,
        index_col=0,
    )  # calcdeltaB(sim,iono,starttime,endtime,station,t0_UT=t0_UT)
    dat = calc_B_vectors(output, station)
elif sys.argv[3] == "EPJ":
    timeseries = iono.import_timerange(starttime, endtime, t0_UT, sys.argv[5])
    ejet = elecproject(station)
    ejet.sample_region(timeseries, sim_coords=False)
    ejet.calc_ground_fields(Z=1e20 + 1e20j, t_window=1 * 60)
    if sys.argv[5] == "GEO":
        dat = {"times": ejet.mag_times[::600], "b_xyz": 1e9 * ejet.B_ground[:, ::600].T}
    else:
        dat = {"times": ejet.mag_times[::600], "b_nez": 1e9 * ejet.B_ground[:, ::600].T}
    times_UT = [
        (ejet.datetimes[0] + dt.timedelta(seconds=int(i)))
        for i in ejet.mag_times[::600] - ejet.mag_times[0]
    ]
    dat["UT"] = times_UT

if sys.argv[5] == "GEO":
    df = pd.DataFrame(
        np.array([dat["b_xyz"][:, 0], dat["b_xyz"][:, 1], dat["b_xyz"][:, 2]]).T,
        index=dat["UT"],
        columns=["bx", "by", "bz"],
    )
else:
    df = pd.DataFrame(
        np.array([dat["b_nez"][:, 0], dat["b_nez"][:, 1], dat["b_nez"][:, 2]]).T,
        index=dat["UT"],
        columns=["bn", "be", "bz"],
    )
df.to_csv(
    path_to_data
    + "/Data/"
    + sys.argv[3]
    + "/output"
    + sys.argv[4]
    + "_"
    + sys.argv[5]
    + ".csv",
    index_label="timestep",
)
