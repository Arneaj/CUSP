"""Functions for visualising solar wind data."""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def plot_SW(gorgon_SW, coords, source=None, show_UT=True, simtime=0, disp=True):
    """Plot a time-series of each of the solar wind variables used for Gorgon input.

    Args:
    ----
        gorgon_SW (pd.DataFrane): Pandas DataFrame from the 'gen_gorgon_SW' function.
        coords (str): Coordinate system out of 'GSE', 'GSM', 'SM' and 'sim'
        corresponding to input data.
        source (str, optional): Solar wind data source to optionally be displayed in the
        title. Defaults to None.
        show_UT (bool, optional): Choose whether to display UT time instead of
        simulation time along the x-axis. Defaults to True.
        simtime (int, optional): Initial simulation time in seconds from which the solar
        wind data will be applied. Defaults to 0.
        disp (bool, optional): Choose whether to display the plot or not.
        Defaults to True.

    Returns:
    -------
        None

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("ticks")

    if show_UT:
        time = gorgon_SW.index  # Use original UT time array
    else:
        d_t = (gorgon_SW.index[1] - gorgon_SW.index[0]).total_seconds()
        time = (
            np.arange(simtime, simtime + d_t * len(gorgon_SW.index), d_t) / 3600
        )  # Use simulation time array

    fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    bs = ["Bx", "By", "Bz", "B"]
    b_labs = [r"$B_x$", r"$B_y$", r"$B_z$", r"$B$"]
    for i in range(len(bs)):
        ax[0].plot(time, gorgon_SW[bs[i]], label=b_labs[i])
        ax[0].set_ylabel(r"$\vec{B}_{" + coords + "}$ (nT)", fontsize=13)

    vs = ["vx", "vy", "vz", "v"]
    v_labs = [r"$v_x$", r"$v_y$", r"$v_z$", r"$v$"]
    for i in range(len(vs)):
        ax[1].plot(time, gorgon_SW[vs[i]], label=v_labs[i])
        ax[1].set_ylabel(r"$\vec{v}_{" + coords + "}$ (km/s)", fontsize=13)

    ax[2].plot(time, gorgon_SW.n, "gray")
    ax[2].set_ylabel(r"$n$ (/cm$^3$)", fontsize=13)

    ax[3].plot(time, gorgon_SW.Ti, "orange")
    ax[3].set_ylabel(r"$T_{i,e}$ (eV)", fontsize=13)

    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlim(time[0], time[-1])

    if source is not None:
        ax[0].set_title(source + " Data", fontsize=12)

    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[1].yaxis.set_major_locator(MultipleLocator(100))
    ax[1].yaxis.set_minor_locator(MultipleLocator(20))
    ax[2].yaxis.set_major_locator(MultipleLocator(5))
    ax[2].yaxis.set_minor_locator(MultipleLocator(1))
    ax[3].yaxis.set_major_locator(
        MultipleLocator(max(10, 20 * (np.max(gorgon_SW.Ti) > 100)))
    )
    ax[3].yaxis.set_minor_locator(
        MultipleLocator(max(5, 10 * (np.max(gorgon_SW.Ti) > 100)))
    )
    i_days = (
        int((gorgon_SW.index[-1] - gorgon_SW.index[0]).total_seconds()) // (3600 * 24)
        + 1
    )

    for axi in ax:
        axi.grid()
        if show_UT:
            axi.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30 * i_days))
            axi.xaxis.set_major_locator(mdates.HourLocator(interval=1 * i_days))
            locator = mdates.AutoDateLocator(minticks=15, maxticks=20)
            formatter = mdates.ConciseDateFormatter(locator)
            axi.xaxis.set_major_formatter(formatter)
        else:
            plt.locator_params(axis="x", nbins=12)  # x-axis

    if show_UT:
        for label in ax[-1].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        plt.savefig(
            "SW_Input_Data_" + coords + "_UT.pdf", format="pdf", bbox_inches="tight"
        )
    else:
        plt.xlabel(r"$T_{sim}$ (h)", fontsize=13)
        plt.savefig(
            "SW_Input_Data_" + coords + ".pdf", format="pdf", bbox_inches="tight"
        )
    if disp:
        plt.show()
    else:
        plt.close()
