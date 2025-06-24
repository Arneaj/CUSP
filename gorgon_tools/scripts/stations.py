"""Module to provide utility functions for working with station data."""

import csv
import datetime as dt
import sys
from os.path import exists

import numpy as np
import pandas as pd

from ..geomagnetic.coordinates import GEO_to_MAG, sph_to_cart, subsolar_angles

r_P = 6.371e6
AE_syn = ["A" + ("%02d" % i) for i in range(0, 48)]
KP_syn = ["K" + ("%02d" % i) for i in range(0, 48)]
grid_syn = ["G" + "%03d" % i for i in range(0, 1000)]


def calc_station_coords(station, times_UT, syn_colat=False):
    """Calculate the cartesian coordinates of a station.

    Args:
    ----
        station (list): A list containing the station name, GEO colatitude, and
        longitude.
        times_UT (list): A list of datetime objects corresponding to the simulation
        times.
        syn_colat (bool): If True, use synthetic colatitude. Defaults to False.

    Returns:
    -------
        tuple: A tuple containing the station name, and the x, y, and z coordinates of
        the station.

    """
    if station[0] in AE_syn or station[0] in KP_syn or station[0] in grid_syn:
        obs_th_SM, obs_az_SM = (
            np.array(station[1]) * np.pi / 180 + np.zeros([len(times_UT)]),
            np.array(station[2]) * np.pi / 180 + np.zeros([len(times_UT)]),
        )
        obs_x, obs_y, obs_z = sph_to_cart(
            r_P, 0 * obs_az_SM + obs_th_SM, obs_az_SM
        )  # to cartesian, adding pi to azimuth for sim x,y directions
    else:
        obs_th_GEO, obs_az_GEO = (
            np.array(station[1:]) * np.pi / 180
        )  # station GEO colatitude, longitude in radians
        obs_lat_MAG, obs_az_MAG = GEO_to_MAG(
            np.pi / 2 - obs_th_GEO, obs_az_GEO, np.array(times_UT)
        )  # GEO to MAG
        obs_th_MAG = np.pi / 2 - obs_lat_MAG  # back to colatitude
        if syn_colat:
            obs_th_MAG = (
                station[1] * np.pi / 180 * np.ones_like(obs_th_MAG)
            )  # if synthetic colatitude is specified, use that instead
        _, ss_az = subsolar_angles(
            np.array(times_UT)
        )  # angle between MAG and SM X-axes
        obs_th_SM, obs_az_SM = obs_th_MAG, obs_az_MAG - ss_az  # MAG to SM
        obs_x, obs_y, obs_z = sph_to_cart(
            r_P, 0 * obs_az_SM + obs_th_SM, obs_az_SM + np.pi
        )  # to cartesian, adding pi to azimuth for sim x,y directions
    obs_name = station[0]

    return obs_name, obs_x, obs_y, obs_z


def create_station_list(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    dlat,
    dlon,
    filename="./gridded_GEO.tsv",
    label="G",
):
    """Automate creating a gridded station list for input.

    lat_min (float): Southernmost colatitude, range from [0,180] degrees.
    lat_max (float): Northernmost colatitude, range from [0,180] degrees.
    lon_min (float): Westernmost longitude, range from [-180,180] degrees.
    lon_max (float): Easternmost longitude, range from [-180,180] degrees.
    dlat (float): Latitude grid resolution in degrees.
    dlon (float): Longitude grid resolution in degrees.
    filename (string): Filename to be outputted to. Defaults to './gridded_GEO.tsv'.
    label (char): Label to associated with gridded data. Defaults to 'G'.
    """
    flag = True
    if abs(lon_min) > 180 or abs(lon_max) > 180:
        print("Longitude out of range, please constrain to [-180,180]")
        flag = False
    if lon_min > lon_max:
        print("Westernmost longitude exceeds easternmost longitude.")
        flag = False
    if lat_min < 0 or lat_max < 0:
        print("Colatitude out of range, please constrain to [0,180]")
        flag = False
    if lat_min > lat_max:
        print("Northernmost latitude exceeds southernmost latitude.")
        flag = False
    if np.mod(lat_max - lat_min, dlat) != 0:
        print(
            "Non-integer splitting of latitude resolution, terminating at nearest "
            "latitude to lat_max."
        )
    if np.mod(lon_max - lon_min, dlon) != 0:
        print(
            "Non-integer splitting of longitude resolution, terminating at nearest "
            "longitude to lon_max."
        )

    if flag is True:
        with open(filename, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            ind = 0
            for lat in np.arange(lat_min, lat_max + dlat, dlat):
                if lat <= lat_max:
                    for lon in np.arange(lon_min, lon_max + dlon, dlon):
                        if lon <= lon_max:
                            writer.writerow([ind, lat, lon, label + str(ind).zfill(3)])
                            ind += 1
    return


if sys.argv[4] == "all":
    stn_lists = ["AE", "KP", "gridded"]
elif sys.argv[4] == "SAGE":
    stn_lists = ["AE", "KP", "SAGE"]
else:
    stn_lists = [sys.argv[4]]
stn_dfs = []
syn_colat = None
for stn_list in stn_lists:
    try:
        stn_dat = pd.read_csv(
            "../data/stations/" + stn_list + ".tsv",
            delim_whitespace=True,
            header=None,
            index_col=0,
        )
        stn_dfs.append(stn_dat)
    except Exception:
        if len(sys.argv) > 5:
            fnm_suffix = "_GEO.tsv"
            if len(sys.argv) > 6:
                syn_colat = sys.argv[6].lower() == "true"
                fnm_suffix = "GEO_SYN_MLAT.tsv"
            if len(sys.argv[5]) > 2:
                stn_dat = pd.read_csv(
                    "../data/stations/" + stn_list + fnm_suffix,
                    delim_whitespace=True,
                    header=None,
                    index_col=0,
                )
                stn_dfs.append(stn_dat)


stations = pd.concat(stn_dfs, axis=0, ignore_index=True)
stations.columns = ["colat", "lon", "stn"]

try:
    times_t0 = int(float(sys.argv[1]))
    times_tN = int(float(sys.argv[2]))
    ground_dumtout = int(float(sys.argv[3]))
    times = np.arange(times_t0, times_tN + ground_dumtout, ground_dumtout)
    t0_UT = dt.datetime.strptime(
        sys.argv[5], "%Y-%m-%d_%H:%M:%S"
    )  # UT time corresponding to zero simulation time
    times_UT = [t0_UT + dt.timedelta(seconds=int(t)) for t in times]
    file_exists = [exists("../data/stations/stns_%d.csv" % (t)) for t in times]
    for i in stations.index:
        obs_name, obs_x, obs_y, obs_z = calc_station_coords(
            [str(stations.loc[i].stn), stations.loc[i].colat, stations.loc[i].lon],
            times_UT,
            syn_colat,
        )
        for t in range(len(times)):
            if not file_exists[t]:
                with open("../data/stations/stns_%d.csv" % (times[t]), "a") as f:
                    if i == 0:
                        f.write(" stn, x_sim, y_sim, z_sim\n")
                    f.write(
                        f"{obs_name:>4}, "
                        f"{obs_x[t]: 1.7e}, "
                        f"{obs_y[t]: 1.7e}, "
                        f"{obs_z[t]: 1.7e}\n"
                    )
except Exception:
    file_exists = exists("../data/stations/stns_sim.csv")
    if not file_exists:
        with open("../data/stations/stns_sim.csv", "a") as f:
            f.write("stn, x_sim, y_sim, z_sim\n")
            for i in stations.index:
                station = [
                    stations.loc[i].stn,
                    stations.loc[i].colat,
                    stations.loc[i].lon,
                ]
                obsi_th, obsi_az = (
                    np.array(station[1:]) * np.pi / 180
                )  # if idealised run just take simulation coords as input for iono
                # and FAC
                obs_x, obs_y, obs_z = sph_to_cart(r_P, obsi_th, obsi_az)
                f.write(
                    f"{station[0]:>4}, {obs_x: 1.7e}, {obs_y: 1.7e}, {obs_z: 1.7e}\n"
                )
