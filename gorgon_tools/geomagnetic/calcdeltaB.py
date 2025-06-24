"""Module provides functions for calculating the ground magnetic field.

This module provides functions for calculating the ground magnetic field due to 
magnetospheric and ionospheric currents using the Biot-Savart law.
"""

import datetime as dt

import numpy as np
import pandas as pd
from scipy.constants import mu_0

from .coordinates import (
    GEI_to_GSE,
    GEO_to_GEI,
    GEO_to_MAG,
    GSE_to_SM,
    cart_to_sph,
    cart_to_sph_vec,
    sph_to_cart,
    sph_to_cart_vec,
    subsolar_angles,
)


def tangent_mask(x, y, z, x_obs, y_obs, z_obs):
    """TODO Write doctstring."""
    return ((x - x_obs) * x_obs + (y - y_obs) * y_obs + (z - z_obs) * z_obs) >= 0


def apply_mask(x, x_mask):
    """TODO Write doctstring."""
    return np.where(x_mask, x, 0.0)


def biot_savart(x_obs, y_obs, z_obs, x, y, z, jx, jy, jz, dV):
    """TODO Write doctstring."""
    # distances must be in km, j must already be masked
    # displacement vectors
    rx = x_obs - x
    ry = y_obs - y
    rz = z_obs - z
    rtot = np.sqrt(rx**2 + ry**2 + rz**2)

    # B-field in T in sim coords
    bx = np.sum((mu_0 / (4.0 * np.pi)) * dV * (jy * rz - jz * ry) / (rtot**3))
    by = np.sum((mu_0 / (4.0 * np.pi)) * dV * (jz * rx - jx * rz) / (rtot**3))
    bz = np.sum((mu_0 / (4.0 * np.pi)) * dV * (jx * ry - jy * rx) / (rtot**3))

    return bx, by, bz


def calcdeltaB(
    sim,
    iono,
    starttime,
    endtime,
    station,
    t0_UT=None,
    dt_skip=1,
    dr_IB=1,
    output_folder=None,
    r_n=400,
    extent="full",
):
    """Perform a global Biot-Savart integration and return B-field at a given station.

    Performs a global Biot-Savart integration of all magnetospheric and ionospheric
    currents in the simulation, returning the B-field at a given station in simulation
    coordinates from each contribution as a pandas dataframe.

    Args:
    ----
        sim (gorgon_sim): Gorgon magnetosphere class.
        iono (ionosphere): Gorgon ionosphere class.
        starttime (int): Initial simulation time in seconds
        from which to do calculations.
        endtime (int): Final simulation time in seconds at which to do calculations.
        station (list): List of length 3 containing station name and geographic
        colatitude, longitude in radians, e.g. ['ESK', 0.605, 6.227].
        t0_UT (datetime,optional): UT time corresponding to t = 0 in the simulation -
        if None, assumes idealised run and does no transformations. Defaults to None.
        dt_skip (int, optional): Timesteps to skip, e.g. dt = 5 uses every 5th timestep.
        Defaults to 1.
        dr_IB (float, optional): Number of grid cells away from the inner boundary at
        which calculation of FACs ends. Defaults to 1.
        output_folder (str, optional): Folder in which to write output file, if None
        then no file will be written. Defaults to None.
        r_n: number of radial layers in gap region.
        extent: extent of simulation box ('full' is entire domain, otherwise list of
        extents in Re from center in the various axes [x,y,z]).

    Returns:
    -------
        (pd.DataFrame): Pandas dataframe containing time-series of cartesian components
        of ground field for each current contribution.

    """
    # This will suppress various numpy warnings as these can be quite verbose
    import warnings as warnings

    warnings.filterwarnings("ignore")

    # Define time range
    MS_times_dump, IS_times_dump = sim.times, iono.times
    MS_time_dump, IS_time_dump = sim.time, iono.time
    if len(sim.times) > len(iono.times):
        sim.times = sim.times[sim.timestep(starttime) : sim.timestep(endtime) + 1][
            ::dt_skip
        ]
        IS_MS_inds = [(IS_t in sim.times) for IS_t in iono.times]
        iono.times == np.array(iono.times)[IS_MS_inds]
        times = sim.times
    else:
        iono.times = iono.times[iono.timestep(starttime) : iono.timestep(endtime) + 1][
            ::dt_skip
        ]
        MS_IS_inds = [(MS_t in iono.times) for MS_t in sim.times]
        sim.times == np.array(sim.times)[MS_IS_inds]
        times = iono.times
    sim.import_timestep(0, ["jvec"])
    iono.import_timestep(0)

    # Create dataframe
    output = pd.DataFrame(
        index=np.array(times).astype(float),
        columns=[
            "Bx_m",
            "Bx_f",
            "Bx_h",
            "Bx_p",
            "By_m",
            "By_f",
            "By_h",
            "By_p",
            "Bz_m",
            "Bz_f",
            "Bz_h",
            "Bz_p",
        ],
    )

    # Get station spherical coords in simulation
    if t0_UT is not None:
        times_UT = [t0_UT + dt.timedelta(seconds=int(t)) for t in times]
        obs_th_GEO, obs_az_GEO = station[1:]  # station GEO colatitude, longitude
        obs_lat_MAG, obs_az_MAG = GEO_to_MAG(
            np.pi / 2 - obs_th_GEO, obs_az_GEO, t0_UT
        )  # GEO to MAG
        obs_th_MAG = np.pi / 2 - obs_lat_MAG  # back to colatitude
        _, ss_az = subsolar_angles(times_UT)  # angle between MAG and SM X-axes
        obs_th_SM, obs_az_SM = obs_th_MAG, obs_az_MAG - ss_az  # MAG to SM
        obs_x, obs_y, obs_z = sph_to_cart(
            iono.r_P, obs_th_SM, obs_az_SM + np.pi
        )  # to cartesian, adding pi to azimuth for sim x,y directions
        _, obsi_th, obsi_az = cart_to_sph(
            obs_x, obs_y, obs_z
        )  # back to spherical for iono and FAC
        obs_x, obs_z = (
            np.cos(-iono.tilt * np.pi / 180) * obs_x
            + np.sin(-iono.tilt * np.pi / 180) * obs_z,
            -np.sin(-iono.tilt * np.pi / 180) * obs_x
            + np.cos(-iono.tilt * np.pi / 180) * obs_z,
        )  # Rotate for magnetosphere
    else:
        obsi_th, obsi_az = station[
            1:
        ]  # if idealised run just take simulation coords as input for iono and FAC
        obs_x, obs_y, obs_z = sph_to_cart(iono.r_P, obsi_th, obsi_az)
        obs_x, obs_z = (
            np.cos(-iono.tilt * np.pi / 180) * obs_x
            + np.sin(-iono.tilt * np.pi / 180) * obs_z,
            -np.sin(-iono.tilt * np.pi / 180) * obs_x
            + np.cos(-iono.tilt * np.pi / 180) * obs_z,
        )  # Rotate for magnetosphere
    obs_name = station[0]

    # define inner boundary
    ib = iono.r_MS + (
        dr_IB * sim.dx[0] * iono.r_P
    )  # define inner boundary such that currents on boundary are excluded
    # (plus one grid cell)

    # grid for magnetosphere (note: jvec output vti is cell-centred)
    if extent == "full":
        x_m, y_m, z_m = np.meshgrid(
            sim.xc * iono.r_P, sim.yc * iono.r_P, sim.zc * iono.r_P, indexing="ij"
        )
    else:
        midx = int(sim.center[0] / sim.dx[0])
        midy = int(sim.center[1] / sim.dx[1])
        midz = int(sim.center[2] / sim.dx[2])
        extx = int(extent[0] / sim.dx[0])
        exty = int(extent[1] / sim.dx[1])
        extz = int(extent[2] / sim.dx[2])
        x_m, y_m, z_m = np.meshgrid(
            sim.xc[midx - extx : midx + extx] * iono.r_P,
            sim.yc[midy - exty : midy + exty] * iono.r_P,
            sim.zc[midz - extz : midz + extz] * iono.r_P,
            indexing="ij",
        )

    # grid for FAC
    r_m = np.linspace(
        iono.r_IS, ib, r_n
    )  # define radial elements for FAC up to inner boundary
    FAC_az, FAC_th, FAC_r = np.meshgrid(
        iono.az, iono.th, r_m, indexing="ij"
    )  # assuming a regular 3D grid
    x_f, y_f, z_f = sph_to_cart(FAC_r, FAC_th, FAC_az)  # convert to cartesian coords

    # grid for ionosphere, rotate into simulation coordinayes
    iono_az, iono_th = np.meshgrid(
        iono.az, iono.th, indexing="ij"
    )  # assuming a regular 2D grid for thin shell
    x_i, y_i, z_i = sph_to_cart(
        iono.r_IS, iono_th, iono_az
    )  # convert to cartesian coords

    # define inner boundary mask for magnetosphere
    ib_mask = (
        np.sqrt(x_m**2 + y_m**2 + z_m**2) > ib
    )  # avoids areas within inner boundary

    # define volume element for magnetosphere
    dx_m, dy_m, dz_m = np.meshgrid(
        sim.dx[midx - extx : midx + extx],
        sim.dy[midy - exty : midy + exty],
        sim.dz[midz - extz : midz + extz],
        indexing="ij",
    )
    dV_m = dx_m * dy_m * dz_m * (iono.r_P**3)

    # grid of ionospheric cell spacings
    dazs, dths = iono.daz_cyc, np.append(iono.dth, iono.dth[-1])
    daz, dth = np.meshgrid(dazs, dths, indexing="ij")
    daz_r, dth_r = np.tile(daz.T, reps=(r_n, 1, 1)), np.tile(dth.T, reps=(r_n, 1, 1))
    daz_r, dth_r = np.swapaxes(daz_r, 0, 2), np.swapaxes(dth_r, 0, 2)

    # define volume element for every FAC filament
    dr = r_m[1] - r_m[0]  # radial element
    dV_f = FAC_r**2 * dr * np.sin(FAC_th) * dth_r * daz_r
    dV_f[:, 0, :] = (
        FAC_r[:, 0, :] ** 2
        * dr
        * np.sin(FAC_th[:, 1, :] / 4.0)
        * (dth_r[:, 1, :] / 2.0)
        * daz_r[:, 1, :]
    )  # correct NP
    dV_f[:, -1, :] = (
        FAC_r[:, -1, :] ** 2
        * dr
        * np.sin(np.pi - ((np.pi - FAC_th[:, -2, :]) / 4.0))
        * (dth_r[:, -2, :] / 2.0)
        * daz_r[:, -2, :]
    )  # correct SP

    # define volume element for ionosphere (actually a surface element on a thin shell)
    dS_i = iono.r_IS**2 * np.sin(iono_th) * dth * daz
    dS_i[:, 0] = (
        iono.r_IS**2 * np.sin(iono_th[:, 1] / 4.0) * (dth[:, 1] / 2.0) * daz[:, 1]
    )  # correct for NP
    dS_i[:, -1] = (
        iono.r_IS**2
        * np.sin(np.pi - ((np.pi - iono_th[:, -2]) / 4.0))
        * (dth[:, -1] / 2.0)
        * daz[:, -2]
    )  # correct for SP

    # loop through simulation timesteps
    for it, t in enumerate(times):
        if t >= sim.times[sim.timestep(sim.time) + 1]:
            sim.import_timestep(sim.timestep(t), ["jvec"])
        if t >= iono.times[iono.timestep(iono.time) + 1]:
            iono.import_timestep(
                iono.timestep(t)
            )  # import corresponding ionosphere timestep

        if extent == "full":
            sim_jvec = sim.arr["jvec"][:, :, :, :]
        else:
            sim_jvec = sim.arr["jvec"][
                midx - extx : midx + extx,
                midy - exty : midy + exty,
                midz - extz : midz + extz,
                :,
            ]

        # get cartesian co-ords for station for timestep
        if t0_UT is not None:
            xi, yi, zi = sph_to_cart(
                iono.r_P, obsi_th[it], obsi_az[it]
            )  # convert to cartesian
            xm, ym, zm = obs_x[it], obs_y[it], obs_z[it]
        else:
            xi, yi, zi = sph_to_cart(iono.r_P, obsi_th, obsi_az)  # convert to cartesian
            xm, ym, zm = obs_x, obs_y, obs_z

        # define tangent mask appropriate to timestep
        tang_mask_m = tangent_mask(x_m, y_m, z_m, xm, ym, zm)
        tang_mask_f = tangent_mask(x_f, y_f, z_f, xi, yi, zi)
        tang_mask_i = tangent_mask(x_i, y_i, z_i, xi, yi, zi)

        ###### MAGNETOSPHERIC CURRENTS
        # read in magnetospheric data from timestep and populate array
        jx_m = sim_jvec[:, :, :, 0]
        jy_m = sim_jvec[:, :, :, 1]
        jz_m = sim_jvec[:, :, :, 2]
        # mask inner boundary and tangent masks
        jx_m = apply_mask(jx_m, ib_mask & tang_mask_m)
        jy_m = apply_mask(jy_m, ib_mask & tang_mask_m)
        jz_m = apply_mask(jz_m, ib_mask & tang_mask_m)
        # get contribution to ground magnetic field
        bx_m, by_m, bz_m = biot_savart(
            xm, ym, zm, x_m, y_m, z_m, jx_m, jy_m, jz_m, dV_m
        )
        # rotate back to original sim
        bx_m, bz_m = (
            np.cos(iono.tilt * np.pi / 180) * bx_m
            + np.sin(iono.tilt * np.pi / 180) * bz_m,
            -np.sin(iono.tilt * np.pi / 180) * bx_m
            + np.cos(iono.tilt * np.pi / 180) * bz_m,
        )

        ###### FIELD-ALIGNED CURRENTS (in gap region)
        # read in FAC data
        J_FAC = np.repeat(
            iono.arr["FAC"].T[:, :, np.newaxis], len(r_m), axis=2
        )  # get ionospheric contribution
        # define vector components along FAC grid in spherical co-ords
        jr_f = (
            -2.0
            * np.cos(FAC_th)
            * J_FAC
            * (iono.r_IS / FAC_r) ** 3
            / np.sqrt(1 + 3.0 * (np.cos(FAC_th) ** 2))
        )
        jaz_f = (
            0.0
            * J_FAC
            * (iono.r_IS / FAC_r) ** 3
            / np.sqrt(1 + 3.0 * (np.cos(FAC_th) ** 2))
        )
        jth_f = (
            -np.sin(FAC_th)
            * J_FAC
            * (iono.r_IS / FAC_r) ** 3
            / np.sqrt(1 + 3.0 * (np.cos(FAC_th) ** 2))
        )
        # convert to cartesian co-ords
        jx_f, jy_f, jz_f = sph_to_cart_vec(jr_f, jth_f, jaz_f, FAC_th, FAC_az)
        # apply tangent mask
        jx_f = apply_mask(jx_f, tang_mask_f)
        jy_f = apply_mask(jy_f, tang_mask_f)
        jz_f = apply_mask(jz_f, tang_mask_f)
        # get contribution to ground magnetic field
        bx_f, by_f, bz_f = biot_savart(
            xi, yi, zi, x_f, y_f, z_f, jx_f, jy_f, jz_f, dV_f
        )

        ###### IONOSPHERIC HORIZONTAL CURRENTS
        if "sig_P" not in iono.arr_names:
            iono.arr["sig_P"] = iono.arr["phi"] * 0.0 + 10.0
        if "sig_H" not in iono.arr_names:
            iono.arr["sig_H"] = iono.arr["phi"] * 0.0 + 0.0

        # read in ionospheric data to get j_perp = sig_P*E + sig_H*(b x E)
        # (where b is assumed to be radial)
        E_az, E_th = np.gradient(
            -iono.arr["phi"].T, iono.az, iono.th, axis=(0, 1)
        )  # get e-field from grad(phi)
        E_th = E_th / iono.r_IS  # scale grad correctly
        E_az[:, 1:-1] = E_az[:, 1:-1] / (iono.r_IS * np.sin(iono_th[:, 1:-1]))
        E_az[:, 0] = np.mean(E_az[:, 1])
        E_az[:, -1] = np.mean(E_az[:, -2])
        hemi_scale = (iono_th >= np.pi / 2.0) * 1.0 - (
            iono_th < np.pi / 2.0
        ) * 1.0  # ensure B-field is in right direction
        # split into hall and pedersen currents
        # hall
        jth_h = -hemi_scale * iono.arr["sig_H"].T * E_az
        jaz_h = hemi_scale * iono.arr["sig_H"].T * E_th
        # pedersen
        jth_p = iono.arr["sig_P"].T * E_th
        jaz_p = iono.arr["sig_P"].T * E_az
        # convert both to cartesian
        jx_h, jy_h, jz_h = sph_to_cart_vec(
            np.zeros_like(jth_h), jth_h, jaz_h, iono_th, iono_az
        )
        jx_p, jy_p, jz_p = sph_to_cart_vec(
            np.zeros_like(jth_p), jth_p, jaz_p, iono_th, iono_az
        )
        # apply tangent masks
        jx_h = apply_mask(jx_h, tang_mask_i)
        jy_h = apply_mask(jy_h, tang_mask_i)
        jz_h = apply_mask(jz_h, tang_mask_i)
        jx_p = apply_mask(jx_p, tang_mask_i)
        jy_p = apply_mask(jy_p, tang_mask_i)
        jz_p = apply_mask(jz_p, tang_mask_i)
        # get contributions to ground magnetic field
        bx_h, by_h, bz_h = biot_savart(
            xi, yi, zi, x_i, y_i, z_i, jx_h, jy_h, jz_h, dS_i
        )
        bx_p, by_p, bz_p = biot_savart(
            xi, yi, zi, x_i, y_i, z_i, jx_p, jy_p, jz_p, dS_i
        )

        # save to output dataframe as nT
        output.loc[t].Bx_m = bx_m * 1e9
        output.loc[t].Bx_f = bx_f * 1e9
        output.loc[t].Bx_h = bx_h * 1e9
        output.loc[t].Bx_p = bx_p * 1e9

        output.loc[t].By_m = by_m * 1e9
        output.loc[t].By_f = by_f * 1e9
        output.loc[t].By_h = by_h * 1e9
        output.loc[t].By_p = by_p * 1e9

        output.loc[t].Bz_m = bz_m * 1e9
        output.loc[t].Bz_f = bz_f * 1e9
        output.loc[t].Bz_h = bz_h * 1e9
        output.loc[t].Bz_p = bz_p * 1e9

    if t0_UT is not None:
        output.insert(0, "UT", times_UT)
    output.insert(1 - 1 * (t0_UT is None), "th", obsi_th)
    output.insert(2 - 1 * (t0_UT is None), "az", obsi_az)
    output.index.name = "timestep"
    if output_folder is not None:
        output.to_csv(
            output_folder + "/output" + obs_name + ".csv"
        )  # write to file with the station name

    sim.times, iono.times = MS_times_dump, IS_times_dump  # restore full time arrays
    sim.time, iono.time = MS_time_dump, IS_time_dump

    return output


def calc_B_vectors(output, station):
    """Calculate B-field vector components.

    Calculates B-field vector components from the calc_deltaB output in XYZ (geographic)
    and NEZ (geomagnetic coordinates) either for a real event or idealised run. Result
    is returned as a dictionary also containing the horizontal (bh) component as well as
    the field magnitude (bmag) and rate of change (db/bt).

    Args:
    ----
        output (pd.DataFrame): Pandas dataframe containing result of
        calc_deltaB function.
        station (list): List of length 3 containing station name and geographic
        colatitude, longitude in radians, e.g. ['ESK', 0.605, 6.227].
        sim_coords (boolean): Flag to only use simulation co-ords
        in a similar way to CIM.

    Returns:
    -------
        (dict): Dictionary containing magnetic field vector time-series, station data
        and time range information.

    """
    # Creating a dictionary which we will use to store the ground field in
    # different coordinates
    station_dat = {}
    station_dat["name"] = station[0]

    # Load dataframe
    output = output.dropna()
    station_dat["times"] = output.index.values
    UT_flag = "UT" in output.keys()

    # Get station simulation coords
    if UT_flag:
        station_dat["UT"] = list(pd.to_datetime(output.UT.values))
        comps = ["nez", "xyz"]
    else:
        comps = ["nez"]
    # Get station coordinates in SM
    station_dat["th"], station_dat["az"] = (
        np.array(output.th).astype(float),
        np.array(output.az).astype(float) + np.pi,
    )

    # If UT time provided, get station coordinates in GEO
    AE_syn = ["A" + ("%02d" % i) for i in range(0, 48)]
    KP_syn = ["K" + ("%02d" % i) for i in range(0, 48)]
    grid_syn = ["G" + "%03d" % i for i in range(0, 1000)]
    if UT_flag:
        if station[0] in AE_syn or station[0] in KP_syn or station[0] in grid_syn:
            obs_x_SM, obs_y_SM, obs_z_SM = sph_to_cart(
                1.0, station_dat["th"], station_dat["az"]
            )
            obs_x_GSE, obs_y_GSE, obs_z_GSE = GSE_to_SM(
                obs_x_SM, obs_y_SM, obs_z_SM, np.array(station_dat["UT"]), inv=True
            )
            obs_x_GEI, obs_y_GEI, obs_z_GEI = GEI_to_GSE(
                obs_x_GSE, obs_y_GSE, obs_z_GSE, station_dat["UT"], inv=True
            )
            obs_x_GEO, obs_y_GEO, obs_z_GEO = GEO_to_GEI(
                obs_x_GEI, obs_y_GEI, obs_z_GEI, station_dat["UT"], inv=True
            )
            _, station_dat["th_GEO"], station_dat["az_GEO"] = cart_to_sph(
                obs_x_GEO, obs_y_GEO, obs_z_GEO
            )
        else:
            station_dat["th_GEO"], station_dat["az_GEO"] = (
                0 * station_dat["th"] + station[1],
                0 * station_dat["th"] + station[2],
            )

    # Initialise dictionary arrays
    station_dat["B_nez"] = np.array(
        [0 * output["Bx_m"], 0 * output["By_m"], 0 * output["Bz_m"]]
    ).T
    if UT_flag:
        station_dat["B_xyz"] = np.array(
            [0 * output["Bx_m"], 0 * output["By_m"], 0 * output["Bz_m"]]
        ).T

    # Loop through each contribution
    for i in ["m", "f", "h", "p"]:
        # NEZ is easy as spherical B components are same in SM and MAG
        bx_SM, by_SM, bz_SM = (
            -output["Bx_" + i].values,
            -output["By_" + i].values,
            output["Bz_" + i].values,
        )
        br, bth, baz = cart_to_sph_vec(
            bx_SM, by_SM, bz_SM, station_dat["th"], station_dat["az"]
        )
        station_dat["B_nez_" + i] = np.array([-bth, baz, -br]).T

        # XYZ require geographic coordinates: only defined if UT time provided, rotate
        # into SM then into GEO...
        if UT_flag:
            bx_GSE, by_GSE, bz_GSE = GSE_to_SM(
                bx_SM, by_SM, bz_SM, np.array(station_dat["UT"]), inv=True
            )
            bx_GEI, by_GEI, bz_GEI = GEI_to_GSE(
                bx_GSE, by_GSE, bz_GSE, station_dat["UT"], inv=True
            )
            bx_GEO, by_GEO, bz_GEO = GEO_to_GEI(
                bx_GEI, by_GEI, bz_GEI, station_dat["UT"], inv=True
            )
            br_GEO, bth_GEO, baz_GEO = cart_to_sph_vec(
                bx_GEO, by_GEO, bz_GEO, station_dat["th_GEO"], station_dat["az_GEO"]
            )
            station_dat["B_xyz_" + i] = np.array([-bth_GEO, baz_GEO, -br_GEO]).T

    # add up different contributions
    for i in ["m", "f", "h", "p"]:
        for comp in comps:
            station_dat["B_" + comp] = (
                station_dat["B_" + comp] + station_dat["B_" + comp + "_" + i]
            )

    return station_dat
