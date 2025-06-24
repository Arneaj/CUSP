"""Module containing functions for generating reports from Gorgon output files."""
import datetime as dt

import imageio
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import seaborn as sns
from IPython.display import Image, display
from matplotlib.ticker import MultipleLocator

from ..geomagnetic.benchmarks import calc_auroral_indices
from ..geomagnetic.coordinates import GSM_to_SM, SM_to_SMD
from ..geomagnetic.dB_import import ground
from ..magnetosphere.gorgon_import import gorgon_sim
from ..magnetosphere.visualise import IB_mask, stretched_to_uniform_2D


def latex(x):
    """Convert a string to LaTeX format."""
    return "$\mathsf{" + x + "}$"


sns.set_context("notebook")
sns.set_palette("colorblind")
sns.set_style("ticks")
default_cmap = "viridis"
plt.rcParams["image.cmap"] = default_cmap


def MP_sub_solar(sim, starttime_UT=None, use_SMD=True):
    """Calculate the subsolar point in the magnetopause (MP) in GSM coordinates.

    Args:
    ----
        sim (gorgon_sim): Simulation object.
        starttime_UT (datetime): UT time corresponding to zero simulation time.
        use_SMD (bool): If True, uses SMD coordinates for calculation.

    Returns:
    -------
        float: Subsolar point in GSM coordinates.

    """
    import datetime as dt

    from ..magnetosphere.connectivity import calc_connectivity

    ixf = np.arange(len(sim.xb))[sim.xb < -4][-1]

    xs = -(sim.xb[ixf::-1] + 0.5 * sim.dx[0])
    x = -np.arange(np.min(xs), np.max(xs), 0.1)
    y, z = 0 * np.ones_like(x), 0 * np.ones_like(x)

    if starttime_UT is not None:
        time = np.array([starttime_UT + dt.timedelta(seconds=int(sim.time))])
        if use_SMD:
            x, y, z, _, _ = SM_to_SMD(-x, -y, z, time, inv=True)
        x, y, z = GSM_to_SM(x, y, z, time, inv=True)
        x, y = -x, -y
    x0 = np.stack([x, y, z]).T

    link, _ = calc_connectivity(
        x0,
        sim.arr["Bvec_c"],
        np.array([sim.dx[0], sim.dy[0], sim.dz[0]]),
        sim.center - np.array([sim.dx[0], sim.dy[0], sim.dz[0]]),
        ns=10000,
    )
    sw = link == 1
    cl = link == 2

    return -(x[sw][np.argmin(abs(x[sw]))] + x[cl][np.argmax(abs(x[cl]))]) / 2


def plot_metrics(
    sim,
    iono,
    rolling=True,
    coords="SMD",
    use_GSM=False,
    starttime=None,
    endtime=None,
    starttime_UT=None,
    gorgon_SW=None,
    disp=True,
    write_txt=False,
    filename=None,
    fileformat="png",
):
    """Plot various metrics for a Gorgon simulation and ionospheric model.

    Args:
    ----
        sim (gorgon_sim): Simulation object.
        iono (ionosphere): Ionospheric model object.
        rolling (bool): If True, plots rolling averages of the metrics.
        coords (str): Coordinate system to use for plotting.
        use_GSM (bool): If True, plots in GSM coordinates.
        starttime (float): Simulation time in seconds
        endtime (float): Simulation time in seconds
        starttime_UT (datetime): UT time corresponding to zero simulation time.
        gorgon_SW (pd.DataFrame): Solar wind data.
        disp (bool): Choose whether to display the plot or not.
        write_txt (bool): Choose whether to write the data to a text file or not.
        filename (str): Name of the file to save the plot to.
        fileformat (str): Format of the file to save the plot to.

    Returns:
    -------
        float: Subsolar point in GSM coordinates.

    """
    from ..ionosphere.analysis import calc_CPCP, calc_TFAC

    if starttime is None:
        starttime = sim.times[0]
    if endtime is None:
        endtime = sim.times[-1]
    MS_times = range(sim.timestep(starttime), sim.timestep(endtime) + 1)
    IS_times = range(iono.timestep(starttime), iono.timestep(endtime) + 1)
    MS_time = sim.times[MS_times]
    IS_time = iono.times[IS_times]

    r_MP, CPCP_N, TFAC_N = (
        np.zeros([len(MS_times)]),
        np.zeros([len(IS_times)]),
        np.zeros([len(IS_times)]),
    )

    for it, t in enumerate(MS_times):
        sim.import_timestep(t, ["Bvec_c"])
        if use_GSM:
            r_MP[it] = MP_sub_solar(sim, starttime_UT, use_SMD=True)
        else:
            r_MP[it] = MP_sub_solar(sim, None)
    for it, t in enumerate(IS_times):
        iono.import_timestep(t, ["FAC", "phi"])
        CPCP_N[it], _ = calc_CPCP(iono)
        TFAC_N[it], _ = calc_TFAC(iono)

    if rolling:
        n = int((MS_time[-1] - MS_time[-2]) / (IS_time[-1] - IS_time[-2]))
        CPCP_N = np.array(
            [
                np.mean(CPCP_N[i - n // 2 : i + n // 2 + 1])
                for i in range(n, len(CPCP_N) - n, n)
            ]
        )
        CPCP_N = np.append(
            np.append([np.mean(CPCP_N[: n // 2 + 1])], CPCP_N),
            [np.mean(CPCP_N[-n // 2 :])],
        )
        TFAC_N = np.array(
            [
                np.mean(TFAC_N[i - n // 2 : i + n // 2 + 1])
                for i in range(n, len(TFAC_N) - n, n)
            ]
        )
        TFAC_N = np.append(
            np.append([np.mean(TFAC_N[: n // 2 + 1])], TFAC_N),
            [np.mean(TFAC_N[-n // 2 :])],
        )

    xp = (~np.isnan(r_MP)).ravel().nonzero()[0]
    fp = r_MP[~np.isnan(r_MP)]
    x = np.isnan(r_MP).ravel().nonzero()[0]

    r_MP[np.isnan(r_MP)] = np.interp(x, xp, fp)

    if starttime_UT is not None:
        MS_time = [
            starttime_UT + dt.timedelta(hours=(float(i) - float(starttime)) / 3600)
            for i in sim.times[MS_times]
        ]
        if rolling:
            IS_time = MS_time
        else:
            IS_time = [
                starttime_UT + dt.timedelta(hours=(float(i) - float(starttime)) / 3600)
                for i in iono.times[IS_times]
            ]
        endtime = (
            starttime_UT
            + dt.timedelta(seconds=int(endtime - starttime))
            + dt.timedelta(minutes=1)
        )
        starttime = starttime_UT - dt.timedelta(minutes=1)
        if gorgon_SW is not None:
            from ..models.magnetopause import MP_Lin2010, MP_Shue1998

            start_str, end_str = (
                starttime.strftime("%Y-%m-%d %H:%M"),
                endtime.strftime("%Y-%m-%d %H:%M"),
            )
            SW_times = gorgon_SW.loc[start_str:end_str].index
            Pd_SW = (
                (gorgon_SW["n"][start_str:end_str] * 1.67e-27 * 1e6)
                * (gorgon_SW["v"][start_str:end_str] * 1e3) ** 2
                * 1e9
            )
            Pb_SW = (
                (gorgon_SW["B"][start_str:end_str] * 1e-9) ** 2 / (2 * 1.257e-6) * 1e9
            )
            Bx_SW = gorgon_SW["Bx"][start_str:end_str]
            By_SW = gorgon_SW["Bx"][start_str:end_str]
            Bz_SW = gorgon_SW["Bz"][start_str:end_str]
            if use_GSM:
                if coords in ["SM", "SMD"]:
                    Bx_SW, By_SW, Bz_SW = GSM_to_SM(
                        -Bx_SW, -By_SW, Bz_SW, np.array([dt_val for dt_val in SW_times])
                    )
                    if coords == "SMD":
                        _, _, Bz_SW, _, _ = SM_to_SMD(
                            Bx_SW,
                            By_SW,
                            Bz_SW,
                            np.array([dt_val for dt_val in SW_times]),
                        )
                else:
                    print(
                        "No valid solar wind coordinates provided; "
                        "assuming GSM Bz input."
                    )
            r_MP_Shue, r_MP_Lin = np.zeros([len(SW_times)]), np.zeros([len(SW_times)])
            for it in range(len(SW_times)):
                r_MP_Shue[it] = MP_Shue1998(0, Pd_SW[SW_times[it]], Bz_SW[SW_times[it]])
                r_MP_Lin[it] = MP_Lin2010(
                    0,
                    0,
                    Pd_SW[SW_times[it]],
                    Pb_SW[SW_times[it]],
                    Bz_SW[SW_times[it]],
                    iono.tilt,
                )

    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("ticks")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(MS_time, r_MP, color="grey")
    if gorgon_SW is not None and starttime_UT is not None:
        ax[0].plot(
            gorgon_SW.loc[start_str:end_str].index,
            r_MP_Shue,
            color="red",
            ls="dashed",
            label="Shue 1998",
        )
        ax[0].plot(
            gorgon_SW.loc[start_str:end_str].index,
            r_MP_Lin,
            color="blue",
            ls="dashed",
            label="Lin 2010",
        )
    ax[0].set_ylabel(r"$R_{MP}$ / $R_E$", fontsize=13)
    ax[0].legend()

    ax[1].plot(IS_time, CPCP_N / 1e3, color="royalblue")
    ax[1].set_ylabel("CPCP / kV", color="royalblue", fontsize=13)
    ax[1].tick_params("y", colors="royalblue")  # ,labelsize=12)

    ax2 = ax[1].twinx()
    ax2.plot(IS_time, TFAC_N / 1e6, color="firebrick")
    ax2.set_ylabel("TFAC / MA", color="firebrick", fontsize=13)
    ax2.tick_params("y", colors="firebrick")  # ,labelsize=12)

    for axi in ax:
        if starttime_UT is None:
            axi.set_xlabel(r"$t$ / s", fontsize=13)
            plt.locator_params(axis="x", nbins=12)  # x-axis
        axi.set_rasterized(True)
        axi.set_xlim(starttime, endtime)
        axi.tick_params("x", labelsize=10)
        axi.grid()
    ax[0].set_ylim(
        0, max(15, 1.1 * np.max([np.max(r_MP), np.max(r_MP_Shue), np.max(r_MP_Lin)]))
    )
    ax[1].set_ylim(0, max(200, 1.1 * np.max(CPCP_N / 1e3)))
    ax2.set_ylim(0, max(10, 1.1 * np.max(TFAC_N / 1e6)))

    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[1].yaxis.set_major_locator(MultipleLocator(50))
    ax[1].yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    if starttime_UT is not None:
        i_days = int((endtime - starttime).total_seconds()) // (3600 * 24) + 1
        for axi in ax:
            axi.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30 * i_days))
            axi.xaxis.set_major_locator(mdates.HourLocator(interval=1 * i_days))
            locator = mdates.AutoDateLocator(minticks=15, maxticks=20)
            formatter = mdates.ConciseDateFormatter(locator)
            axi.xaxis.set_major_formatter(formatter)
            for label in axi.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)

    plt.tight_layout()

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    plt.show()
    if write_txt:
        IS_MS_inds = [(IS_t in MS_time) for IS_t in IS_time]
        np.savetxt(
            filename[:-4] + ".txt",
            np.array(
                [
                    [t.strftime("%Y-%m-%d %H:%M:%S") for t in MS_time],
                    np.round(r_MP, 2),
                    np.round(r_MP_Shue, 2),
                    np.round(r_MP_Lin, 2),
                    np.round(CPCP_N[IS_MS_inds] / 1e3, 2),
                    np.round(TFAC_N[IS_MS_inds] / 1e6, 2),
                ]
            ).T,
            delimiter=",",
            fmt="%s",
            header=(
                "time (s), R_MP (R_E), Shue1998 (R_E), "
                "Lin2012 (R_E), CPCP (kV), TFAC (MA)"
            ),
        )


def calc_GSM(sim, starttime, coords="SMD", GSM_vars=None):
    """Calculate the subsolar point in the magnetopause (MP) in GSM coordinates.

    Interpolates simulation data from the simulation coordinate system to the Geocentric
    Solar Magnetospheric (GSM) coordinate system.

    Args:
    ----
        sim (gorgon_sim): Simulation object.
        starttime (datetime): UT time corresponding to zero simulation time.
        coords (str): Coordinate system of simulation data.
        GSM_vars (list): List of simulation variables to interpolate.

    Returns:
    -------
        float: Subsolar point in GSM coordinates.

    """
    time = starttime + dt.timedelta(seconds=int(sim.time))
    if coords == "SMD":
        _, _, _, _, del_mu = SM_to_SMD(0, 0, 1, time, "SMD")
        ang = del_mu
    else:  # assume SM coords
        Mx, My, Mz = GSM_to_SM(0, 0, 1, time, inv=True)
        mu = np.arctan(Mx / np.sqrt(My**2 + Mz**2))
        ang = mu
    y_min, y_max = -31, 31
    z_min, z_max = -31, 31
    x_min, x_max = (
        np.squeeze(np.ceil((sim.xb[0] + np.sin(ang) * z_max) / np.cos(ang) / sim.dx[0]))
        * sim.dx[0]
        - 1,
        41,
    )
    sim.xb_GSM = -np.arange(x_min, x_max + 1.1 * sim.dx[-1], sim.dx[0])
    sim.yb_GSM = -np.arange(y_min, y_max + 1.1 * sim.dy[-1], sim.dy[0])
    sim.zb_GSM = np.arange(z_min, z_max + 1.1 * sim.dz[-1], sim.dz[0])
    sim.xc_GSM, sim.yc_GSM, sim.zc_GSM = (
        sim.xb_GSM[:-1] + 0.5 * sim.dx[0],
        sim.yb_GSM[:-1] + 0.5 * sim.dy[0],
        sim.zb_GSM[:-1] + 0.5 * sim.dz[0],
    )
    x_GSM, y_GSM, z_GSM = np.meshgrid(sim.xc_GSM, sim.yc_GSM, sim.zc_GSM, indexing="ij")
    sim.center_GSM = np.array(
        [
            abs(sim.xb_GSM[0]) / sim.dx[0],
            abs(sim.yb_GSM[0]) / sim.dy[0],
            abs(sim.zb_GSM[0]) / sim.dz[0],
        ]
    )

    if coords in ["SM", "SMD"]:
        x_SM, y_SM, z_SM = GSM_to_SM(x_GSM, y_GSM, z_GSM, time)
        if coords == "SMD":
            x_SMD, y_SMD, z_SMD, _, _ = SM_to_SMD(x_SM, y_SM, z_SM, time)
            x_sim, y_sim, z_sim = -x_SMD, -y_SMD, z_SMD
        else:
            x_sim, y_sim, z_sim = -x_SM, -y_SM, z_SM
    xs = x_sim.ravel()
    ys = y_sim.ravel()
    zs = z_sim.ravel()

    if GSM_vars is None:
        GSM_vars = sim.arr_names
    for var in GSM_vars:
        int_3D = interp.RegularGridInterpolator(
            (sim.xc, sim.yc, sim.zc), sim.arr[var], bounds_error=False
        )
        if len(sim.arr[var].shape) > 3:
            sim.arr[var + "_GSM"] = int_3D(np.stack([xs, ys, zs]).T).reshape(
                [len(sim.xc_GSM), len(sim.yc_GSM), len(sim.zc_GSM), 3]
            )
            var_x, var_y, var_z = (
                sim.arr[var + "_GSM"][:, :, :, 0],
                sim.arr[var + "_GSM"][:, :, :, 1],
                sim.arr[var + "_GSM"][:, :, :, 2],
            )
            if coords == "SMD":  # else assume SM coords
                var_x, var_y, var_z, _, _ = SM_to_SMD(
                    var_x, var_y, var_z, time, inv=True
                )
            var_x, var_y, var_z = GSM_to_SM(var_x, var_y, var_z, time, inv=True)
            (
                sim.arr[var + "_GSM"][:, :, :, 0],
                sim.arr[var + "_GSM"][:, :, :, 1],
                sim.arr[var + "_GSM"][:, :, :, 2],
            ) = (-var_x, -var_y, var_z)
        else:
            sim.arr[var + "_GSM"] = int_3D(np.stack([xs, ys, zs]).T).reshape(
                x_sim.shape
            )
        sim.arr[var + "_GSM"] = np.nan_to_num(sim.arr[var + "_GSM"])


def clean_import(df):
    """Todo: Docstring for clean_import."""
    if df.time.dtype == object:
        df.drop(
            df[df.time.str.contains("time")].index, inplace=True
        )  # Drops rows that contain additional headers
        for i in (
            df.columns
        ):  # Convert object data (due to import with headers) into numeric data
            try:
                df[i] = pd.to_numeric(df[i])
            except Exception as e:
                print("Error in converting column to numeric type:", i)
                print(e)
                pass
        # Assuming monotonically increasing time, throw away overlapping timesteps due
        # to restarts, with most recent favoured
        tmp = df.iloc[-1].time  # Start with the last time instance
        for i in range(len(df) - 1, 0, -1):
            if (df.iloc[i].time < tmp) & (
                df.iloc[i - 1].time < df.iloc[i].time
            ):  # If montonically increased and not more than a future time, pass
                pass
            elif (
                df.iloc[i - 1].time > df.iloc[i].time
            ):  # If not monotonically increasing, must be a restart discontinuity
                tmp = df.iloc[
                    i
                ].time  # Update time instance, will drop in next iteration
            elif df.iloc[i].time > tmp:
                df.drop(df.index[i], inplace=True)
        # To avoid cases where same restart timesteps are used at different points but
        # avoid interating again, add condition that difference should be > 0.5ms
        df.drop(df[(abs(df.time.diff(periods=-1)) < 5e-4)].index, inplace=True)
    return df


def import_xy(fdir, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for import_xy."""
    xy = pd.read_csv(fdir, delimiter=",", header=0)
    xy.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # Removes whitespace from column names
    xy = xy.iloc[:, :-1]
    xy = clean_import(xy)
    xy.index = t_fun[0](xy["time"])

    xy["Tot_E"] = xy[["tten", "tket", "ttbe", "ttfus", "ttradloss"]].sum(axis=1)
    xy["Tot_W"] = xy[
        ["work_ele", "work_ion", "work_visc", "work_kdebar", "work_ohm"]
    ].sum(axis=1)

    return xy


def plot_quick_look(xy, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_quick_look."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))

    xy.plot(y="dt", ax=ax[0])
    ax[0].legend(["Tot. dt"])
    ax[0].set(ylabel=latex("\Delta t\;(s)"), title="Timestep")

    xy.plot(y="Tot_E", ax=ax[1])
    ax[1].legend(["Tot. E"])
    ax[1].set(ylabel=latex("E\;(J)"), title="Total Energy")

    xy.plot(y="Tot_W", ax=ax[2])
    ax[2].legend(["Tot. W"])
    ax[2].set(ylabel=latex("W\;(J/m)"), title="Total Work")

    xy.plot(y="qtmt", ax=ax[3])
    ax[3].legend(["Tot. Mass"])
    ax[3].set(ylabel=latex("M\;(kg)"), title="Total Mass")

    ax[-1].set_xlabel("Time " + t_fun[1])
    ax[-1].set_xlim(xy.index[0] // 60, xy.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_energy(xy, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_energy."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
    s_list = ["tten", "tket", "ttbe", "ttfus", "ttradloss"]
    leg = ["Int.", "Kin.", "Mag.", "Fus.", "Rad. Loss"]

    for s, label in zip(s_list, leg):
        ax[0].semilogy(xy.index, xy[s], label=label)
    ax[0].semilogy(xy.index, xy["Tot_E"], ":k", label="Tot. E")
    ax[0].set(title="Energy")
    ax[0].legend(ncol=3)

    for s in s_list[:2]:
        ax[1].plot(xy.index, xy[s])
    ax[1].legend(leg[:-1])
    ax[1].set(title="Internal and Kinetic Energy")

    ax[2].plot(xy.index, xy["ttbe"])
    ax[2].legend(["Mag."])
    ax[2].set(title="Magnetic Energy")

    for i, s in enumerate(s_list[3:]):
        ax[3].plot(xy.index, xy[s], color="C" + str(i + 4), label=s)
    ax[3].legend()
    ax[3].set(title="Fusion and Radiation Losses")

    ax[-1].set_xlabel("Time " + t_fun[1])
    ax[-1].set_xlim(xy.index[0] // 60, xy.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_work(xy, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_work."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    s_list = ["work_ele", "work_ion", "work_visc", "work_kdebar", "Tot_W"]  #
    leg = ["Ele.", "Ion", "Visc.", "Kin. deBar", "Total"]
    xy[s_list].plot(ax=ax[0])
    ax[0].legend(leg)
    ax[0].set(title="Work")
    xy["work_kdebar"].plot(ax=ax[1])
    ax[1].legend("Kin. deBar")
    ax[1].set(title="deBar Correction")

    ax[-1].set_xlabel("Time " + t_fun[1])
    ax[-1].set_xlim(xy.index[0] // 60, xy.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_timestep(xy, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_timestep."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5))

    xy.plot(y=["dt_fluid", "dt_mhd", "dt_rad", "dt_cs", "dt"], ax=ax[0])

    ax[0].legend(["Fluid", "MHD", "Rad.", r"$c_s$", "Total"])
    ax[0].set(ylabel=latex("\Delta t\;(s)"), title="Timestep", yscale="log")

    xy.plot(y="isub_mhd", ax=ax[1])  # , fmt='.')
    MAX = xy["isub_mhd"].max()
    ax[1].set(ylabel=latex("i_{MHD}"), yticks=np.arange(0, MAX + 2, int(MAX // 10) + 1))

    ax[-1].set_xlabel("Time " + t_fun[1])
    ax[-1].set_xlim(xy.index[0] // 60, xy.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_real_time_comparision(
    sim_dir, xy, t_unit, array_name="rho", prefix="00", include_xy=False
):
    """Todo: Docstring for plot_real_time_comparision."""
    import glob
    import os

    sns.set_style("ticks")
    # Compare time stamps for real time
    files = glob.glob(sim_dir + "/MS/" + "*" + array_name + "*.pvtr")
    if include_xy:
        files.append(sim_dir + "xy" + prefix + ".csv")
    creation_times = np.sort(np.array([os.path.getmtime(path) for path in files]))
    real_time = np.array(
        [ct - creation_times[0] for ct in creation_times]
    )  # In seconds

    # Look at file timesteps for sim time
    sim = gorgon_sim(sim_dir)
    sim_time = np.array([float(st) for st in sim.times])
    if include_xy:
        sim_time = np.hstack([sim_time, xy["time"].iloc[-1]])
    real_time = real_time[: len(sim_time)]

    real_time = t_unit[0](real_time)
    sim_time = t_unit[0](sim_time)

    # Linear fit
    fit_coeff = np.polyfit(sim_time[-5:], real_time[-5:], 1)

    # Plot
    lims = np.array([sim_time.min(), sim_time.max()])

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(sim_time, real_time, ".-")
    ax[0].plot(lims, np.polyval(fit_coeff, lims))
    ax[0].plot(lims, lims, "--k")
    ax[0].set(
        ylabel="Real Time " + t_unit[1],
        title=(f"Simulation ~{fit_coeff[0]:4.2f}x slower than real time"),
    )
    ax[1].plot(
        0.5 * (sim_time[1:] + sim_time[:-1]),
        np.diff(real_time) / np.diff(sim_time),
        ".-",
    )
    ax[1].axhline(1, linestyle="--", color="k")
    ax[1].set(
        ylabel=latex("\\Delta t_{real}/\\Delta t_{sim}"),
        xlabel="Sim. Time " + t_unit[1],
    )
    # ax[1].set(xlim=lims,)
    # fig.tight_layout()

    ax[-1].set_xlim(xy.index[0] // 60, xy.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def import_IS_metrics(fdir, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for import_IS_metrics."""
    IS = pd.read_csv(fdir, delimiter=",", header=0)
    IS.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # Removes whitespace from column names
    IS = IS.iloc[:, :-1]
    IS = clean_import(IS)
    IS.index = t_fun[0](IS["time"])

    IS["CPCP_N"] /= 1e3
    IS["CPCP_S"] /= 1e3
    IS["TFAC_N"] /= 1e6
    IS["TFAC_S"] /= 1e6
    IS["JH_tot_N"] /= 1e9
    IS["JH_tot_S"] /= 1e9

    return IS


def import_MS_metrics(fdir, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for import_MS_metrics."""
    MS = pd.read_csv(fdir, delimiter=",", header=0)
    MS.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # Removes whitespace from column names
    MS = MS.iloc[:, :-1]
    MS = clean_import(MS)
    MS.index = t_fun[0](MS["time"])

    MS["SW_n"] = MS["SW_rho"] / (1.67e-27 * 1e6)
    MS["SW_vx"] *= 1e-3
    MS["SW_vy"] *= 1e-3
    MS["SW_vz"] *= 1e-3
    MS["SW_v"] = np.sqrt(MS["SW_vx"] ** 2 + MS["SW_vy"] ** 2 + MS["SW_vz"] ** 2)
    MS["SW_bx"] *= 1e9
    MS["SW_by"] *= 1e9
    MS["SW_bz"] *= 1e9
    MS["SW_b"] = np.sqrt(MS["SW_bx"] ** 2 + MS["SW_by"] ** 2 + MS["SW_bz"] ** 2)
    MS["x_MP"] *= -1
    MS["x_BS"] *= -1
    MS["x_MP"][MS["x_MP"] < 0] = np.nan
    MS["x_BS"][MS["x_BS"] < 0] = np.nan
    for x in ("15", "25"):
        MS["b_B" + x + "."] = (
            np.sqrt(
                MS["bx_B" + x + "."] ** 2
                + MS["bx_B" + x + "."] ** 2
                + MS["bx_B" + x + "."] ** 2
            )
            / 1e-9
        )
    for x in ("12", "15", "20", "25"):
        MS["PSP" + x + "."] = MS["PSP" + x + "."] / 1e-9

    return MS


def import_GSO_vars(fdir, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for import_GSO_vars."""
    from ..geomagnetic.coordinates import cart_to_sph_vec

    GSO_pos = pd.read_csv(fdir + "/GSO/GSO_pos.csv", header=0)
    GSO_pos.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # Removes whitespace from column names
    GSO_pos = GSO_pos.iloc[:, :-1]

    if GSO_pos.dtypes[0] == object:
        for i in range(len(GSO_pos)):
            if not GSO_pos.iloc[-i, 0].strip().isnumeric():
                break
        GSO_pos = GSO_pos.iloc[-i + 1 :]
        for i in GSO_pos.columns:
            try:
                GSO_pos[i] = pd.to_numeric(GSO_pos[i])
            except Exception as e:
                print("Error in converting column:", i)
                print(e)
                pass

    n_GSO = len(GSO_pos["az"])

    GSO_i = pd.read_csv(fdir + "/GSO/GSO.0.csv", usecols=[0, 1], header=0)
    GSO_i.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # Removes whitespace from column names
    GSO_i = GSO_i.iloc[:, :-1]
    GSO_i = clean_import(GSO_i)
    GSO = {"time": np.array(t_fun[0](GSO_i["time"])), "az": np.array(GSO_pos["az"])}
    GSO["az"] = np.roll(GSO["az"], len(GSO["az"]) // 2)
    GSO["az"] = np.append(GSO["az"], [180])

    GSO_vars = ["n", "Ti", "Te", "v", "E_r", "E_az", "B"]
    for var in GSO_vars:
        GSO[var] = np.zeros([len(GSO["time"]), len(GSO["az"])])

    i_azs = [int(i * n_GSO / 360) for i in GSO["az"][:-1]]
    i_azs.append(n_GSO // 2)
    for j, i in enumerate(i_azs):
        GSO_i = pd.read_csv(fdir + "/GSO/GSO." + str(i) + ".csv", header=0)
        GSO_i.rename(
            columns=lambda x: x.strip(), inplace=True
        )  # Removes whitespace from column names
        GSO_i = GSO_i.iloc[:, :-1]
        GSO_i = clean_import(GSO_i)
        GSO_i["n"] = GSO_i["rho"] / (1.67e-27 * 1e6)
        GSO_i["v"] = (
            np.sqrt(GSO_i["vx"] ** 2 + GSO_i["vy"] ** 2 + GSO_i["vz"] ** 2) / 1e3
        )
        GSO_i["E_r"], _, GSO_i["E_az"] = cart_to_sph_vec(
            GSO_i["Ex"] * 1e3,
            GSO_i["Ey"] * 1e3,
            GSO_i["Ez"] * 1e3,
            np.pi / 2,
            GSO["az"][j] * np.pi / 180,
        )
        GSO_i["B"] = (
            np.sqrt(GSO_i["Bx"] ** 2 + GSO_i["By"] ** 2 + GSO_i["Bz"] ** 2) * 1e9
        )
        for var in GSO_vars:
            GSO[var][:, j] = np.array(GSO_i[var])

    return GSO


def import_dB_metrics(fdir, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for import_dB_metrics."""
    dB = ground(fdir)
    dB.import_timeseries(dB.stations)
    inds = calc_auroral_indices(dB, method="inline")
    inds.index = t_fun[0](inds.index)

    return dB, inds


def plot_IS_metrics(IS, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_IS_metrics."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))

    IS.plot(y="CPCP_N", ax=ax[0])
    IS.plot(y="CPCP_S", ax=ax[0])
    ax[0].legend(["North", "South"])
    ax[0].set(ylabel="CPCP (kV)", title="Ionospheric Parameters")

    IS.plot(y="TFAC_N", ax=ax[1])
    IS.plot(y="TFAC_S", ax=ax[1])
    ax[1].legend(["North", "South"])
    ax[1].set(ylabel="TFAC (MA)")

    IS.plot(y="r_PC_N", ax=ax[2])
    IS.plot(y="r_PC_S", ax=ax[2])
    ax[2].legend(["North", "South"])
    ax[2].set(ylabel=r"Polar Cap Radius ($^\circ$)")

    IS.plot(y="JH_tot_N", ax=ax[3])
    IS.plot(y="JH_tot_S", ax=ax[3])
    ax[3].legend(["North", "South"])
    ax[3].set(ylabel="Joule Heating (GW)")

    ax[-1].set_xlabel("Time " + t_fun[1], fontsize=12)
    ax[-1].set_xlim(IS.index[0], IS.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_MS_metrics(MS, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_MS_metrics."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7.5))

    MS.plot(y="x_MP", ax=ax[0], title="Magnetospheric Parameters")
    MS.plot(y="x_BS", ax=ax[0])
    ax[0].legend(["MP", "BS"])
    ax[0].set(ylabel=r"Standoff Distance ($R_E$)")

    MS.plot(y="b_B15.", ax=ax[1])
    MS.plot(y="b_B25.", ax=ax[1])
    ax[1].legend([r"15 $R_E$", r"25 $R_E$"])
    ax[1].set(ylabel=r"Lobe $B$ (nT)")

    MS.plot(y="PSP12.", ax=ax[2])
    MS.plot(y="PSP15.", ax=ax[2])
    MS.plot(y="PSP20.", ax=ax[2])
    MS.plot(y="PSP25.", ax=ax[2])
    ax[2].legend([r"12 $R_E$", r"15 $R_E$", r"20 $R_E$", r"25 $R_E$"])
    ax[2].set(ylabel=r"Plasma Sheet $P$ (nPa)")

    ax[-1].set_xlabel("Time " + t_fun[1], fontsize=12)
    ax[-1].set_xlim(MS.index[0], MS.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_SW_metrics(SW, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_SW_metrics."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 10))

    SW.plot(y="SW_n", ax=ax[0])
    ax[0].legend([])
    ax[0].set(ylabel=r"Number Density (cm$^{-3}$)", title="Solar Wind Parameters")

    SW.plot(y="SW_Ti", ax=ax[1])
    SW.plot(y="SW_Te", ax=ax[1])
    ax[1].legend([r"$T_i$", r"$T_e$"])
    ax[1].set(ylabel=r"Temperature (eV)")

    SW.plot(y="SW_vx", ax=ax[2])
    SW.plot(y="SW_vy", ax=ax[2])
    SW.plot(y="SW_vz", ax=ax[2])
    SW.plot(y="SW_v", ax=ax[2])
    ax[2].legend([r"$v_x$", r"$v_y$", r"$v_z$", r"$v$"])
    ax[2].set(ylabel=r"Velocity (kms$^{-1}$)")

    SW.plot(y="SW_bx", ax=ax[3])
    SW.plot(y="SW_by", ax=ax[3])
    SW.plot(y="SW_bz", ax=ax[3])
    SW.plot(y="SW_b", ax=ax[3])
    ax[3].legend([r"$B_x$", r"$B_y$", r"$B_z$", r"$B$"])
    ax[3].set(ylabel="IMF (nT)")

    ax[-1].set_xlabel("Time " + t_fun[1], fontsize=12)
    ax[-1].set_xlim(SW.index[0], SW.index[-1])
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_GSO_vars(GSO, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_GSO_vars."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(9, 12.5))

    time, az = np.meshgrid(GSO["time"], GSO["az"], indexing="ij")
    az = az * 24 / 360
    az = np.where(az >= 12, az - 12, az + 12)
    az[:, -1] = 24
    col = ax[0].contourf(time, az, GSO["n"], 50, cmap="plasma", zorder=0)
    ax[0].set(
        ylabel="MLT",
        yticks=[0, 6, 12, 18, 24],
        yticklabels=["12", "18", "00", "06", "12"],
        title="GSO Parameters",
    )
    cbar = plt.colorbar(col, ax=ax[0])
    cbar.ax.set_ylabel(r"Number Density (cm$^{-3}$)")

    col = ax[1].contourf(time, az, GSO["Ti"], 50, cmap="hot", zorder=0)
    ax[1].set(
        ylabel="MLT",
        yticks=[0, 6, 12, 18, 24],
        yticklabels=["12", "18", "00", "06", "12"],
    )
    cbar = plt.colorbar(col, ax=ax[1])
    cbar.ax.set_ylabel(r"Ion Temperature (eV)")

    col = ax[2].contourf(time, az, GSO["v"], 50, cmap="PuBu", zorder=0)
    ax[2].set(
        ylabel="MLT",
        yticks=[0, 6, 12, 18, 24],
        yticklabels=["12", "18", "00", "06", "12"],
    )
    cbar = plt.colorbar(col, ax=ax[2])
    cbar.ax.set_ylabel(r"Bulk Velocity (kms$^{-1}$)")

    col = ax[3].contourf(time, az, GSO["E_az"], 50, cmap="viridis", zorder=0)
    ax[3].set(
        ylabel="MLT",
        yticks=[0, 6, 12, 18, 24],
        yticklabels=["12", "18", "00", "06", "12"],
    )
    cbar = plt.colorbar(col, ax=ax[3])
    cbar.ax.set_ylabel(r"Azim. E-field (mVm$^{-1}$)")

    col = ax[4].contourf(time, az, GSO["B"], 50, cmap="bwr", zorder=0)
    ax[4].set(
        ylabel="MLT",
        yticks=[0, 6, 12, 18, 24],
        yticklabels=["12", "18", "00", "06", "12"],
    )
    cbar = plt.colorbar(col, ax=ax[4])
    cbar.ax.set_ylabel("B-Field (nT)")

    col = ax[-1].set_xlabel("Time " + t_fun[1], fontsize=12)
    ax[-1].set_xlim(GSO["time"][0], GSO["time"][-1])

    for axi in ax:
        axi.grid(alpha=0.5)

    plt.tight_layout()

    return fig, ax


def plot_ground_metrics(inds, t_fun=[lambda t: t / 60, "(min)"]):
    """Todo: Docstring for plot_ground_metrics."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    inds.plot(y="AU", ax=ax[0], color="r", label="AU", title="Geomagnetic Indices")
    inds.plot(y="AL", ax=ax[0], color="royalblue", label="AL")
    inds.plot(y="AE", ax=ax[1], color="green", label="AE")
    inds.plot(y="AO", ax=ax[1], color="orange", label="AO")
    ax[-1].set_xlabel("Time " + t_fun[1], fontsize=12)
    ax[-1].set_xlim(inds.index[0], inds.index[-1])
    for axi in ax:
        axi.set_ylabel(r"$B_n$ / nT")
        axi.grid()
        axi.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    return fig, ax


def plot_advection(sim):
    """Todo: Docstring for plot_advection."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9, 3))

    for t in range(0, 301, 50):
        sim.import_timestep(sim.timestep(t))
        ax.plot(sim.xc + 0.5, sim.arr["rho"].squeeze(), label=str(sim.time))

    ax.set_ylabel(r"rho / kgm$^{-3}$")
    ax.set_xlabel("$X$ / m")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(0, 1)
    ax.grid()
    ax.legend(frameon=True, framealpha=0.6, title="Time / ms")
    plt.tight_layout()

    return fig, ax


def plot_Sod_shock_tube(sim):
    """Todo: Docstring for plot_Sod_shock_tube."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7.5))

    sim.import_timestep(-1)
    for var in sim.arr_names:
        sim.arr[var] = np.squeeze(sim.arr[var])

    ax[0].plot(sim.xc, sim.arr["rho"], "grey")
    ax[0].set_ylabel(r"$\rho$ / kgm$^{-3}$")
    ax[0].set_ylim(-0.1, 1.1)

    ax[1].plot(sim.xc, sim.arr["array_pres"], "firebrick")
    ax[1].set_ylabel(r"$P$ / Pa")
    ax[1].set_ylim(-0.1, 1.1)

    ax[2].plot(sim.xc, np.linalg.norm(sim.arr["vvec"], axis=1))
    ax[2].set_ylabel(r"$v$ / ms$^{-1}$")

    ax[-1].set_xlabel("$X$ / m")
    ax[-1].set_xlim(0, 1)
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_Brio_Wu_shock_tube(sim):
    """Todo: Docstring for plot_Brio_Wu_shock_tube."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7.5))

    sim.import_timestep(-1)
    for var in sim.arr_names:
        sim.arr[var] = np.squeeze(sim.arr[var])

    ax[0].plot(sim.xc, sim.arr["rho"], "grey", label=r"$\rho$ / kgm$^{-3}$")
    ax[0].plot(sim.xc, sim.arr["array_pres"], "firebrick", label=r"$P$ / Pa")
    ax[0].legend(frameon=True, framealpha=0.6)
    ax[0].set_ylim(-0.1, 1.1)

    ax[1].plot(sim.xc, sim.arr["vvec"][:, 0], label="x")
    ax[1].plot(sim.xc, sim.arr["vvec"][:, 1], label="y")
    ax[1].plot(sim.xc, sim.arr["vvec"][:, 2], label="z")
    ax[1].plot(sim.xc, np.linalg.norm(sim.arr["vvec"], axis=1), label="Mag.")
    ax[1].set_ylabel(r"$\vec{v}$ / ms$^{-1}$")
    ax[1].legend(frameon=True, framealpha=0.6)

    ax[2].plot(sim.xc, sim.arr["Bvec_c"][:, 0] / np.sqrt(1.26e-7), label="x")
    ax[2].plot(sim.xc, sim.arr["Bvec_c"][:, 1] / np.sqrt(1.26e-7), label="y")
    ax[2].plot(sim.xc, sim.arr["Bvec_c"][:, 2] / np.sqrt(1.26e-7), label="z")
    ax[2].plot(
        sim.xc,
        np.linalg.norm(sim.arr["Bvec_c"], axis=1) / np.sqrt(1.26e-7),
        label="Mag.",
    )
    ax[2].set_ylabel(r"$\vec{B}$ / $\sqrt{\mu_0}$T")
    ax[2].legend(frameon=True, framealpha=0.6)

    ax[-1].set_xlabel("$X$ / m")
    ax[-1].set_xlim(0, 1)
    for axi in ax:
        axi.grid()
    plt.tight_layout()

    return fig, ax


def plot_Orszag_Tang(sim):
    """Todo: Docstring for plot_Orszag_Tang."""
    sns.set_style("ticks")
    import matplotlib.patheffects as PathEffects

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 8.8))

    # sim.xc = sim.xc-0.5
    x, y = np.meshgrid(sim.xc, sim.yc, indexing="ij")
    for i, time in enumerate(range(0, 1001, 500)):
        sim.import_timestep(sim.timestep(time))
        for var in sim.arr_names:
            sim.arr[var] = np.squeeze(sim.arr[var])
        ax[i, 0].contourf(x, y, sim.arr["rho"], 100, cmap="plasma")
        ax[i, 1].contourf(
            x, y, np.linalg.norm(sim.arr["vvec"], axis=2), 100, cmap="viridis"
        )
        ax[i, 2].contourf(
            x, y, np.linalg.norm(sim.arr["Bvec_c"], axis=2), 100, cmap="bwr"
        )
        if i == 0 and np.min(sim.dx) == np.max(sim.dx):
            ax[i, 1].streamplot(
                sim.xc,
                sim.yc,
                -sim.arr["vvec"][:, :, 1],
                -sim.arr["vvec"][:, :, 0],
                color="lightgrey",
            )
            ax[i, 2].streamplot(
                sim.xc,
                sim.yc,
                -sim.arr["Bvec_c"][:, :, 1],
                -sim.arr["Bvec_c"][:, :, 0],
                color="lightgrey",
            )
        for j in range(3):
            txt = ax[i, j].text(
                0.05,
                0.9,
                str(sim.time / 1e3) + " s",
                fontsize=12,
                fontfamily="monospace",
                color="k",
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="w")])

    for i in range(3):
        ax[2, i].set_xlabel(r"$X$ / m")
        ax[2, i].set_xlim(0, 1)
        ax[i, 0].set_ylabel(r"$Y$ / m")
        ax[i, 0].set_ylim(0, 1)

    ax[0, 0].set_title(r"$\rho$")  # / kgm$^{-3}$')
    ax[0, 1].set_title(r"$v$")  # / ms$^{-1}$')
    ax[0, 2].set_title(r"$B$")  # / $\sqrt{\mu_0}$T')

    plt.tight_layout()

    return fig, ax


def plot_Orszag_Tang_cut(sim):
    """Todo: Docstring for plot_Orszag_Tang_cut."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9, 3))

    sim.import_timestep(sim.timestep(500))
    iy = np.argmin(abs(sim.yc - 0.4277))
    ax.plot(sim.xc, sim.arr["array_pres"][:, iy, :].squeeze(), label="Gorgon Test")

    from io import BytesIO
    from pkgutil import get_data

    data = get_data(__name__, "data/Landrillo_and_Zanna.csv")
    dat = np.genfromtxt(BytesIO(data), delimiter=",")
    x, y = dat[:, 0], dat[:, 1]
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    ax.plot(x, y, label="Londrillo and Zanna (2000)")

    ax.set_ylabel(r"$P$ / T")
    ax.set_xlabel("$X$ / m")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(0, 1)
    ax.grid()
    ax.legend(frameon=True, framealpha=0.6)
    plt.tight_layout()

    return fig, ax


def plot_Magneto2D(sim, disp=True, folder=None):
    """Todo: Docstring for plot_Magneto2D."""
    sim.import_timestep(sim.timestep(1800))
    sim.arr["P"] = sim.arr["rho"] / 1.67e-27 * sim.arr["Ti"] * 1.6022e-19
    sim.arr["j"] = np.linalg.norm(sim.arr["jvec"], axis=3)
    if np.min(sim.dx) != np.max(sim.dx):
        stretched_to_uniform_2D(sim, ["P", "j", "Bvec_c"])
        sim.arr["Bvec_c"] = np.reshape(
            sim.arr["Bvec_c"], [len(sim.xc), 1, len(sim.zc), 3]
        )

    # Plotting parameters
    plt_list = [
        [
            "P",
            {
                "name": r"$P$",
                "unit": r"nPa",
                "norm": 1e-9,
                "log": True,
                "min": 1e-5,
                "max": 1e2,
                "cmap": "RdYlBu_r",
            },
        ],
        [
            "j",
            {
                "name": r"$j$",
                "unit": r"nAm$^{-2}$",
                "norm": 1e-9,
                "min": 1e-2,
                "max": 1e2,
                "log": True,
                "cmap": "GnBu_r",
            },
        ],
    ]

    for plts in plt_list:
        if folder is None:
            filename = None
        else:
            filename = folder + "/" + plts[0] + "_" + str(sim.time) + ".png"
        plot_slice_2D(sim, plts, 4, plot_flines=True, disp=disp, filename=filename)


def plot_slice_2D(
    sim, plt_list, r_IB=4, plot_flines=True, disp=True, filename=None, fileformat="png"
):
    """Todo: Docstring for plot_slice_2D."""
    import matplotlib.patheffects as PathEffects
    from matplotlib.colors import LogNorm

    var, params = plt_list

    if "name" in params:
        name = params["name"]
    else:
        name = ""

    if "unit" in params:
        unitlabel = " / " + params["unit"]
    else:
        unitlabel = ""

    if "norm" in params:
        norm = params["norm"]
    else:
        norm = 1.0

    if "min" in params:
        vmin = params["min"]
    else:
        vmin = None

    if "max" in params:
        vmax = params["max"]
    else:
        vmax = None

    if "cmap" in params:
        cmap = params["cmap"]
    else:
        cmap = "RdBu_r"

    if "log" in params:
        log = params["log"]

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")

    fig, ax = plt.subplots(1, 1, figsize=(5.25, 5))

    x, z = sim.xc, sim.zc
    xyz_sgn = np.array([1, 1])

    arr_xz = np.squeeze(sim.arr[var]) / norm
    z_xz, x_xz = np.meshgrid(z, x, indexing="ij")
    z_xz, x_xz = z_xz.T, x_xz.T

    r = np.sqrt(x_xz**2 + z_xz**2)
    mask_xz = r <= r_IB
    coords = [x_xz, z_xz]

    strm_vars = "Bvec_c"
    strm_center = sim.center

    strm_cols = "k"
    seed_pts = np.array(
        [
            np.linspace(-22, 30.1, 10) * xyz_sgn[0],
            np.linspace(-30, 30.1, 10) * xyz_sgn[1],
        ]
    )

    v = arr_xz
    if vmax is None:
        vmax = np.max(v)
    if vmin is None:
        vmin = np.min(v)
    if vmin < 0 and vmax > 0:
        vmin, vmax = -np.max(np.abs([vmin, vmax])), np.max(np.abs([vmin, vmax]))

    if log:
        min_exp, max_exp = np.floor(np.log10(vmin)), np.log10(vmax)
        min_drawn_value, max_drawn_value = (
            1.000001 * 10.0**min_exp,
            0.999999 * 10.0**max_exp,
        )
        levs = 10.0 ** np.arange(
            min_exp, max_exp, (max_exp - min_exp) / 100
        )  # np.power(10, levs_exp)
        v_masked = np.where(v < 10.0**min_exp, min_drawn_value, v)
        v_masked = np.where(v_masked > 10.0**max_exp, max_drawn_value, v_masked)
    else:
        levs = np.arange(1.05 * vmin, 1.05 * vmax, 1.05 * (vmax - vmin) / 100)
        v_masked = 1 * v

    v_masked[mask_xz] = vmin / 10

    if log:
        p = ax.contourf(
            coords[0],
            coords[1],
            v_masked,
            levs,
            norm=LogNorm(vmin=min_drawn_value, vmax=max_drawn_value),
            extend="both",
            cmap=cmap,
        )
    else:
        p = ax.contourf(coords[0], coords[1], v_masked, levs, extend="both", cmap=cmap)

    if plot_flines:
        from ..magnetosphere.streamline import streamline_array

        xstrs, zstrs = seed_pts
        xstr, zstr = np.meshgrid(xstrs, zstrs)
        xstr, zstr = xstr.T, zstr.T
        xs = np.vstack([xstr.ravel(), 0 * zstr.ravel(), zstr.ravel()]).T

        streams = streamline_array(10000, 0.1 * np.mean(sim.dx), direction=0)
        strm_arr = sim.arr[strm_vars].copy()
        strm_arr[:, :, :, 1] = 0
        streams.calc(
            xs,
            strm_arr,
            np.ones([3]) * np.mean(sim.dx),
            strm_center - 0.5 * np.mean(sim.dx),
            v_name=strm_vars,
        )
        for pts in streams.xs:
            ax.plot(pts[:, 0], pts[:, 2], strm_cols, zorder=1, lw=0.7, alpha=0.7)

    ax.set_facecolor("k")
    for c in p.collections:
        c.set_edgecolor("face")

    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlim(-22 * xyz_sgn[0], 40 * xyz_sgn[0])
    ax.set_ylim(-30 * xyz_sgn[1], 30 * xyz_sgn[1])

    txt = ax.text(
        -20.0 * xyz_sgn[0],
        26.0 * xyz_sgn[1],
        f"t = {sim.time} s",
        fontsize=15,
        color="k",
        fontfamily="monospace",
    )
    del_mu = 0
    mu = 0

    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="w")])
    d_arrow = -0.98 * r_IB * np.cos(del_mu), 0, 0.98 * r_IB * np.sin(del_mu)
    ax.arrow(
        0,
        0,
        d_arrow[0],
        d_arrow[1],
        color="w",
        length_includes_head=True,
        head_width=1,
        head_length=1,
        zorder=2,
    )
    d_arrow = -0.98 * r_IB * np.sin(mu), 0, 0.98 * r_IB * np.cos(mu)
    ax.arrow(
        0,
        0,
        d_arrow[0],
        d_arrow[1],
        color="b",
        length_includes_head=True,
        head_width=0.5,
        head_length=0.5,
        lw=1.5,
        zorder=2,
    )
    d_arrow = 0.98 * r_IB * np.sin(mu), 0, -0.98 * r_IB * np.cos(mu)
    ax.arrow(
        0,
        0,
        d_arrow[0],
        d_arrow[1],
        color="r",
        length_includes_head=True,
        head_width=0.5,
        head_length=0.5,
        lw=1.5,
        zorder=2,
    )

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.set_rasterized(True)

    IB_mask(r_IB, ax=ax, color="k", zangle=90 - del_mu * 180 / np.pi)

    ax.set_ylabel(r"$Z$ / $R_E$", fontsize=15, labelpad=-2)
    ax.set_xlabel(r"$X$ / $R_E$", fontsize=15)

    plt.tight_layout()

    cbar_ax = fig.add_axes([1, 0.108, 0.026, 0.864])

    cbar = plt.colorbar(p, cax=cbar_ax)
    if log:
        cbar.set_ticks(10 ** np.arange(min_exp, max_exp + 1))
    cbar.ax.tick_params(axis="both", which="major", labelsize=10)
    cbar.ax.set_ylabel(name + unitlabel, fontsize=14)

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()


def make_gif(folder, labels, descriptor, duration=1, format="png", disp=True):
    """Make a gif from a series of images in a folder.

    A function to create (and optionally display) a gif generated from pre-existing
    source images.

    Args:
    ----
        folder: a string, path to directory containing source images
        labels: a list, containing the unique components of file names, in order of
        frame; should be at start of file name
        descriptor: a string, the remaining part of the file name shared by all the
        files; must end with file extension
        duration: the desired period between each frame in seconds; defaults to 1s
        format: the file type of the images, e.g. 'jpg', defaults to 'png'
        e.g. for folder = 'My_Frames/', labels = [1,2,3,4,5], descriptor = '_Frame.png',
        paths would be 'My_Frames/1_Frame.png' up to 'My_Frames/5_Frame.png'
        disp: a boolean, whether to display the gif in the notebook; defaults to True

    """
    filenames = []
    images = []
    for label in labels:
        filenames.append(folder + str(label) + descriptor)
    for filename in filenames:
        images.append(imageio.imread(filename))
    gifname = "Animation"
    imageio.mimsave(
        folder + descriptor[:-4] + "_" + gifname + "_" + "%.1f" % duration + "s.gif",
        images,
        format="GIF",
        duration=duration,
    )
    if disp:
        with open(
            folder
            + descriptor[:-4]
            + "_"
            + gifname
            + "_"
            + "%.1f" % duration
            + "s.gif",
            "rb",
        ) as f:
            display(Image(data=f.read(), format=format))


def export_to_html(ipynb_name, output_name):
    """Todo: Docstring for export_to_html."""
    import codecs

    import nbformat
    from nbconvert import HTMLExporter

    exporter = HTMLExporter()
    output_notebook = nbformat.read(ipynb_name, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_name, "w", encoding="utf-8").write(output)
