"""Module providing functions for calculating auroral indices.

This module provides functions for calculating auroral indices based on ground field 
data from a Gorgon simulation.
"""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .calcdeltaB import calc_B_vectors
from .coordinates import get_station_coords


def calc_auroral_indices(
    dB,
    method="inline",
    sim=None,
    iono=None,
    AE_GEO=False,
    t0_UT=None,
    starttime=None,
    endtime=None,
    read_existing=False,
    data_folder=None,
    write=False,
    disp=False,
):
    """Calculate ground field at each of the auroral indices stations.

    Calculates the ground field at each of the auroral indices stations as defined at
    http://isgi.unistra.fr/indices_ae.php for a given simulation time range, and
    computes the resulting indices based on the outputs. The user can specify which
    ground field calculation method to use from either 'calcdeltaB' or 'elecproject', as
    well as a folder in which to store the data for each station as well as the indices.
    Note that if existing data for any of the stations is found in said folder
    (with naming format 'output[STN].csv'), then these will be loaded instead of
    recalculating the time series. This allows for a batch job script for producing the
    output prior to calling this function, since the calculation is very slow. If this
    is the case for all stations, the required arguments to the function
    will be ignored.

    Args:
    ----
        dB (calcdeltaB): The calcdeltaB object.
        method (str, optional): Ground field calculation method to use.
        Defaults to 'calcdeltaB'.
        sim (gorgon_sim): Gorgon magnetosphere class.
        iono (ionosphere): Gorgon ionosphere class.
        starttime (int): Initial simulation time in seconds
        from which to do calculations.
        AE_GEO (bool, optional): Choose how to load station names
        endtime (int): Final simulation time in seconds at which to do calculations.
        t0_UT (datetime,optional): UT time corresponding to t = 0 in the simulation -
        if None, assumes idealised run and does no transformations.
        read_existing (bool, optional): Choose whether to read in any output files if
        they already exist in data_folder, if False recalculate data. Defaults to True.
        data_folder (str, optional): Folder in which to locate existing station data and
        write output files, if None then will recalculate data
        and no files will be written. Defaults to None.
        disp (bool, optional): Choose whether to display the plot in case you just wish
        to save the output. Defaults to True.
        write (bool, optional): Choose whether to write outut files in case you just
        wish to display the output. Defaults to True.

    Returns:
    -------
        (pd.DataFrame): Pandas dataframe containing time-series of cartesian components
        of ground field for each current contribution.

    """
    # Load AE station names
    if AE_GEO:
        from io import BytesIO
        from pkgutil import get_data

        data = get_data(__name__, "data/auroral_stations.tsv")
        ae_stations = np.genfromtxt(BytesIO(data), dtype=str)
        station_names = ae_stations[:, 3]
        write_suff = ""
    else:
        station_names = ["A" + ("%02d" % i) for i in range(0, 24)]
        write_suff = "_syn"

    # Loop through and calculate B-field time series at each station
    if read_existing:
        dB.read_timeseries(station_names, data_folder)
    else:
        if method == "inline":
            dB.import_timeseries(station_names, starttime, endtime)
        elif method == "calcdeltaB":
            dB.calcdeltaB(sim, iono, station_names, starttime, endtime)
        elif method == "elecproject":
            dB.elecproject(iono, station_names, starttime, endtime)
        if write:
            dB.write_timeseries(station_names, data_folder)

    ae_bn = pd.DataFrame()
    for i, stn in enumerate(station_names):
        if AE_GEO:
            station = [
                stn,
                float(ae_stations[i, 1]) * np.pi / 180,
                float(ae_stations[i, 2]) * np.pi / 180,
            ]
        else:
            station = [stn, get_station_coords(stn)[0], get_station_coords(stn)[1]]
        output = dB.timeseries[stn]
        dat = calc_B_vectors(output, station)
        ae_bn.insert(
            i, stn, dat["B_nez"][:, 0] - 0
        )  # rough and ready baseline subtraction (assuming start time is
        # geomagnetically quiet), store in dataframe

    # calculate indices from combined dataframe
    if t0_UT is not None:
        ae_bn.index = dat["UT"]
    else:
        ae_bn.index = dat["times"] / 60
    AU = ae_bn.max(axis=1)
    AL = ae_bn.min(axis=1)
    AE = AU - AL
    AO = (AU + AL) / 2

    # Plot Bn for all stations plus AU and AL
    plt.figure()
    for stn in station_names:
        plt.plot(ae_bn[stn])
    plt.plot(AU, "k", linewidth=2.0)
    plt.plot(AL, "k", linewidth=2.0)
    if t0_UT is not None:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel("time / min")
    plt.ylabel("Bn [nT]")

    df = pd.DataFrame(
        np.array([AU, AL, AE, AO]).T,
        index=dat["times"],
        columns=["AU", "AL", "AE", "AO"],
    )
    if t0_UT is not None:
        df.insert(0, "UT", dat["UT"])

    if write:
        plt.savefig(
            data_folder + "Auroral_Indices" + write_suff + ".png", bbox_inches="tight"
        )
        df.to_csv(
            data_folder + "Auroral_Indices" + write_suff + ".txt",
            index_label="timestep",
        )

    if disp:
        plt.show()
    else:
        plt.close()

    if df is not None:
        return df


def plot_benchmarks(
    iono,
    sim,
    starttime,
    endtime,
    t0_UT,
    data_folder,
    plot_CPCP=True,
    rolling=False,
    plot_FPC=True,
    wdc_data=None,
    disp=True,
    filename="Gorgon_Benchmarks.png",
    fileformat="png",
):
    """Plot benchmarks for the Gorgon simulation.

    Args:
    ----
        iono (ionosphere): The ionosphere object.
        sim (gorgon_sim): The Gorgon simulation object.
        starttime (int): The start time of the simulation in seconds.
        endtime (int): The end time of the simulation in seconds.
        t0_UT (datetime): The UT time corresponding to t = 0 in the simulation.
        data_folder (str): The folder in which to locate existing station data and write
        output files.
        plot_CPCP (bool, optional): Whether to plot the CPCP over time.
        Defaults to True.
        rolling (bool, optional): Whether to plot rolling averages. Defaults to False.
        plot_FPC (bool, optional): Whether to plot the FPC over time. Defaults to True.
        wdc_data (str, optional): The path to the WDC data file. Defaults to None.
        disp (bool, optional): Whether to display the plot. Defaults to True.
        filename (str, optional): The name of the file to save the plot to. Defaults to
        'Gorgon_Benchmarks.png'.
        fileformat (str, optional): The format to save the plot in. Defaults to 'png'.

    """
    # get CPCP over time
    if plot_CPCP:
        from gorgon_tools.ionosphere.analysis import calc_CPCP

        CPCPs = []
        CPCP_times = iono.times[iono.timestep(starttime) : iono.timestep(endtime)]
        for t in CPCP_times:
            iono.import_timestep(iono.timestep(t), ["phi"])
            CPCP_N, _ = calc_CPCP(iono)
            CPCPs.append(CPCP_N)
        if t0_UT is not None:
            CPCP_times = [t0_UT + dt.timedelta(seconds=int(t)) for t in CPCP_times]

    if rolling:
        MS_times = range(sim.timestep(starttime), sim.timestep(endtime) + 1)
        IS_times = range(iono.timestep(starttime), iono.timestep(endtime) + 1)
        MS_time = sim.times[MS_times]
        IS_time = iono.times[IS_times]
        n = int((MS_time[-1] - MS_time[-2]) / (IS_time[-1] - IS_time[-2]))
        CPCPs = np.array(
            [
                np.mean(CPCPs[i - n // 2 : i + n // 2 + 1])
                for i in range(n, len(CPCPs) - n, n)
            ]
        )
        CPCPs = np.append(
            np.append([np.mean(CPCPs[: n // 2 + 1])], CPCPs),
            [np.mean(CPCPs[-n // 2 :])],
        )
        CPCP_times = MS_time
        if t0_UT is not None:
            CPCP_times = [t0_UT + dt.timedelta(seconds=int(t)) for t in CPCP_times]

    if plot_FPC:
        from gorgon_tools.ionosphere.analysis import calc_FPC

        FPCs = []
        FPC_times = sim.times[sim.timestep(starttime) : sim.timestep(endtime)]
        for t in FPC_times:
            sim.import_timestep(sim.timestep(t), ["Bvec_c"])
            FPC, _ = calc_FPC(sim, n_mask=4, disp=False)
            FPCs.append(FPC * iono.r_P**2 / 1e9)
        if t0_UT is not None:
            FPC_times = [t0_UT + dt.timedelta(seconds=int(t)) for t in FPC_times]

    # untested, needs generalising but leaving here for future use
    if wdc_data is not None:
        cad = 10  # cadence in mins, either 1 or 10
        indices = np.genfromtxt(data_folder + wdc_data, dtype=str)
        # dt_str_format = "%y%m%dU%HAU"
        AU, AL, AE, AO = (
            np.zeros([25 * 60 // cad]),
            np.zeros([25 * 60 // cad]),
            np.zeros([25 * 60 // cad]),
            np.zeros([25 * 60 // cad]),
        )
        for h in range(25):
            for m in range(60 // cad):
                AU[h * 60 // cad + m] = indices[h, 3 + cad * m]
                AL[h * 60 // cad + m] = indices[25 + h, 3 + cad * m]
                AE[h * 60 // cad + m] = indices[50 + h, 3 + cad * m]
                AO[h * 60 // cad + m] = indices[75 + h, 3 + cad * m]
        obs_times = [
            t0_UT + dt.timedelta(seconds=starttime + 60 * cad * i)
            for i in range(len(AU))
        ]
        obs_indices = pd.DataFrame(np.array([AU, AL, AE, AO]).T)
        obs_indices.index = obs_times
        obs_indices.columns = ["AU", "AL", "AE", "AO"]

    # read-in simulation indices
    sim_indices = pd.read_csv(data_folder + "Auroral_Indices.txt")
    if "UT" in sim_indices.columns:
        sim_indices.index = pd.to_datetime(sim_indices["UT"])

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    sns.set_style("ticks")

    i = 2 - np.sum([plot_CPCP, plot_FPC])
    _, ax = plt.subplots(4 - i, 1, figsize=(9, 7 - i), sharex=True)

    if plot_CPCP or plot_FPC:
        if plot_CPCP:
            ax[0].plot(CPCP_times, np.array(CPCPs) / 1e3, color="k", label="Gorgon")
            ax[0].set_ylabel("CPCP / kV", color="k", fontsize=14)
            ax[0].set_ylim(0, 1.2 * np.max(np.array(CPCPs) / 1e3))
            if plot_FPC:
                ax[1].plot(FPC_times, np.array(FPCs), color="grey", label="Gorgon")
                ax[1].set_ylabel(r"$F_{PC}$ / GWb", color="k", fontsize=14)
                ax[1].set_ylim(0, 1.2 * np.max(np.array(FPCs)))
            ax[0].set_xlim(CPCP_times[0], CPCP_times[-1])
        elif plot_FPC:
            ax[0].plot(FPC_times, np.array(FPCs) * 1e6, color="k", label="Gorgon")
            ax[0].set_ylabel(r"$F_{PC}$ / GWb", color="k", fontsize=14)
            ax[0].set_ylim(0, 1.2 * np.max(np.array(FPCs) * 1e6))
            ax[0].set_xlim(FPC_times[0], FPC_times[-1])
        ax[0].tick_params("y", colors="k", labelsize=12)
        ax[0].tick_params("x", labelsize=13)
    else:
        ax[0].set_xlim(sim_indices.index[0], sim_indices.index[-1])

    ax[2 - i].plot(sim_indices.index, sim_indices["AU"], color="r", label="Gorgon AU")
    ax[2 - i].plot(
        sim_indices.index, sim_indices["AL"], color="royalblue", label="Gorgon AL"
    )
    ax[3 - i].plot(
        sim_indices.index, sim_indices["AE"], color="green", label="Gorgon AE"
    )
    ax[3 - i].plot(
        sim_indices.index, sim_indices["AO"], color="orange", label="Gorgon AO"
    )

    if wdc_data is not None:
        ax[2 - i].plot(
            obs_indices.index, obs_indices["AU"], color="r", ls="--", label="AU"
        )
        ax[2 - i].plot(
            obs_indices.index, obs_indices["AL"], color="royalblue", ls="--", label="AL"
        )
        ax[3 - i].plot(
            obs_indices.index, obs_indices["AE"], color="green", ls="--", label="AE"
        )
        ax[3 - i].plot(
            obs_indices.index, obs_indices["AO"], color="orange", ls="--", label="AO"
        )

    ax[2 - i].set_ylabel(r"$B_n$ / nT", color="k", fontsize=14)
    ax[3 - i].set_ylabel(r"$B_n$ / nT", color="k", fontsize=14)
    ax[2 - i].tick_params("y", colors="k", labelsize=12)
    ax[2 - i].tick_params("x", labelsize=13)
    ax[3 - i].tick_params("y", colors="k", labelsize=12)
    ax[3 - i].tick_params("x", labelsize=13)

    ax[0].xaxis.set_major_locator(ticker.LinearLocator(20))
    locator = mdates.AutoDateLocator(minticks=6, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(formatter)

    for axi in ax:
        axi.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axi.grid()

    plt.tight_layout()

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(data_folder + filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()
