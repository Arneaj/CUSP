"""Module containing functions for visualising geomagnetic data."""
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .coordinates import GEO_to_MAG


def plot_B(
    dB,
    stations,
    coords="MAG",
    starttime=None,
    endtime=None,
    disp=True,
    filename=None,
    fileformat=None,
):
    """Plot the components of the ground magnetic field as a time series.

    Args:
    ----
        dB (dict): Dictionary containing the magnetic field timeseries data.
        stations (list): List of station names to plot.
        coords (str, optional): Coordinate system to plot the magnetic field in.
        Defaults to 'MAG'.
        starttime (int, optional): Start time in seconds to plot. Defaults to None.
        endtime (int, optional): End time in seconds to plot. Defaults to None.
        disp (bool, optional): Choose whether to display the plot in case you just to
        save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    sns.set_style("ticks")

    N_ax = 3
    fig, axs = plt.subplots(N_ax, 1, figsize=(7, 7), sharex=True)

    if coords == "GEO" and dB.t0_UT is None:
        raise RuntimeError(
            "Ininitial UT time 'dB.t0_UT' undefined; geographic xyz "
            "coordinates cannot be determined."
        )

    for stn in stations:
        UT_flag = dB.t0_UT is not None
        if UT_flag:
            times = np.array(pd.to_datetime(dB.timeseries[stn]["UT"]))
            if starttime is not None:
                t0 = times[dB.timeseries[stn].index == starttime]
            else:
                t0 = times[0]
            if endtime is not None:
                t1 = times[dB.timeseries[stn].index == endtime]
            else:
                t1 = times[-1]
        else:
            times = np.array(dB.timeseries[stn].index) / 60
            if starttime is not None:
                t0 = starttime
            else:
                t0 = times[0]
            if endtime is not None:
                t1 = endtime
            else:
                t1 = times[-1]

        Bx, By, Bz = (
            dB.timeseries[stn]["Bx_" + coords],
            dB.timeseries[stn]["By_" + coords],
            dB.timeseries[stn]["Bz_" + coords],
        )
        axs[0].plot(times, Bx)
        axs[1].plot(times, By, label=stn)
        axs[2].plot(times, Bz)

    if coords == "MAG":
        axs[0].set_ylabel(r"$B_n$ / nT", fontsize=14)
        axs[1].set_ylabel(r"$B_e$ / nT", fontsize=14)
    else:
        axs[0].set_ylabel(r"$B_x$ / nT", fontsize=14)
        axs[1].set_ylabel(r"$B_y$ / nT", fontsize=14)
    axs[2].set_ylabel(r"$B_z$ / nT", fontsize=14)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=11)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axs:
        # ax.tick_params(which='both',axis='both',direction='in',top=True,
        # right=True,labelsize=11)
        ax.set_xlim(t0, t1)
        ax.grid()
        if UT_flag:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

    if not UT_flag:
        plt.xlabel("Time / min", fontsize=14)

    plt.tight_layout()

    axs[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()


def plot_global(
    timeseries, plt_list, time=None, disp=True, filename=None, fileformat="png"
):
    """Plot contours of a chosen ionospheric variable over a global geographic map.

    Args:
    ----
        timeseries (dict): Ionospheric timeseries dictionary generated using the
        import_timerange function in the ionosphere module.
        plt_list (dict, optional): Dictionary of plotting parameters
        (see example notebook).
        time (int, optional): Simulation time in seconds to plot. Defaults to None.
        disp (bool, optional): Choose whether to display the plot in case you just to
        save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    # Extracting user input and setting plotting parameters
    var, params = plt_list

    if time is not None:
        t = np.where(timeseries["times"] == time)[0][0]
    else:
        t, time = -1, timeseries["times"][-1]

    coords = timeseries["coords"]
    if coords in ["GEO", "MAG"]:
        arr = timeseries[var + "_" + coords][:, :, t]
    elif coords == "sim":
        arr = timeseries[var][:, :, t]

    if "station" in params:
        from .coordinates import get_station_coords

        th0, az0 = get_station_coords(params["station"])
        if az0 >= np.pi:
            az0 -= 2 * np.pi
        lat0, long0 = 180 / np.pi * np.array([np.pi / 2 - th0, az0])

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
        if np.any(arr < 0):
            vmin = -np.max(abs(arr) / norm)
        else:
            vmin = np.max(arr / norm) * 0.01

    if "max" in params:
        vmax = params["max"]
    else:
        vmax = np.max(abs(arr) / norm)

    if "cmap" in params:
        cmap = params["cmap"]
    else:
        cmap = "RdBu_r"

    if "datetimes" in timeseries.keys():
        UT_time = timeseries["datetimes"][t]
    else:
        UT_time = None
    if coords == "MAG":
        from scipy import interpolate

        thmesh, azmesh = np.swapaxes(
            np.meshgrid(timeseries["th"], timeseries["az"]), 1, 2
        )
        thmesh_GEO, azmesh_GEO = GEO_to_MAG(
            np.pi / 2 - thmesh, azmesh, UT_time, inv=True
        )
        arr = interpolate.griddata(
            np.array([np.pi / 2 - thmesh_GEO.ravel(), azmesh_GEO.ravel()]).T,
            arr.ravel(),
            (thmesh, azmesh),
            method="cubic",
            fill_value=0,
        )
        arr[:, 0] = 0.5 * (
            arr[:, -1] + arr[:, 1]
        )  # linear average around zero longitude to fill values

    # Cyclical azimuthal boundary
    from cartopy.util import add_cyclic_point
    from scipy import interpolate

    # ensure ionospheric array is uniform
    uni_az = np.linspace(0, 2 * np.pi, len(timeseries["az"]))
    uni_th = np.linspace(0, np.pi, len(timeseries["th"]))

    uni_thmesh, uni_azmesh = np.swapaxes(np.meshgrid(uni_th, uni_az), 1, 2)
    arr = interpolate.griddata(
        np.array([uni_thmesh.ravel(), uni_azmesh.ravel()]).T,
        arr.ravel(),
        (uni_thmesh, uni_azmesh),
        method="cubic",
        fill_value=0,
    )
    arr[:, 0] = 0.5 * (
        arr[:, -1] + arr[:, 1]
    )  # linear average around zero longitude to fill values

    arr, lons = add_cyclic_point(arr, coord=uni_az)
    thmesh, azmesh = np.swapaxes(np.meshgrid(np.pi / 2 - uni_th, lons), 1, 2)
    thmesh, azmesh = 180 / np.pi * thmesh, 180 / np.pi * azmesh
    levs = np.arange(vmin, vmax * 1.01, (vmax - vmin) / 11)

    # Plot contour over map background
    import cartopy.crs as ccrs

    # fig = plt.figure(figsize=(17, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    col1 = ax.contour(
        azmesh,
        thmesh,
        arr / norm,
        levs,
        zorder=1,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        alpha=1,
    )
    cbar1 = plt.colorbar(col1, ax=ax)
    cbar1.ax.set_ylabel(name + unitlabel)
    ax.coastlines(color="grey")
    ax.stock_img()

    # Optionally overlay location of geomagnetic station
    if "station" in params:
        import matplotlib.patheffects as PathEffects

        ax.scatter(long0, lat0, color="r", s=40, edgecolors="w", zorder=2)
        txt1 = ax.text(
            long0 + 4, lat0, params["station"], size=12, zorder=2, color="black"
        )
        txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    if UT_time is not None:
        plt.title(UT_time)
    else:
        plt.title(r"$t$ = " + str(time))

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()


def plot_local(ejet, plt_list, time=None, disp=True, filename=None, fileformat="png"):
    """Plot the electrojet, ionoionospheric electric potential in the electrojet region.

    Args:
    ----
        ejet (eletrojet): Electrojet object containing ionospheric data within a given
        region.
        plt_list (dict, optional): Dictionary of plotting parameters
        (see example notebook).
        time (int, optional): Simulation time in seconds to plot. Defaults to None.
        disp (bool, optional): Choose whether to display the plot in case you just to
        save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    import cartopy.crs as ccrs
    import matplotlib.patheffects as PathEffects

    # Extracting user input and setting plotting parameters
    var, params = plt_list

    if time is not None:
        t = np.where(ejet.times == time)[0][0]
    else:
        t, time = -1, ejet.times[-1]

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

    # Extract spatial data
    th0, az0 = ejet.loc
    if az0 >= np.pi:
        az0 -= 2 * np.pi
    lat0, long0 = 180 / np.pi * np.array([np.pi / 2 - th0, az0])

    # Extract variable data, transform into GEO if sample region is in MAG
    if ejet.datetimes is not None:
        UT_time = ejet.datetimes[t]
    else:
        UT_time = None

    th, az = np.meshgrid(
        np.pi / 2 - ejet.iono_sample["th"], ejet.iono_sample["az"], indexing="ij"
    )
    if ejet.coords == "MAG":
        th, az = GEO_to_MAG(th, az, UT_time, inv=True)
    th, az = th * 180 / np.pi, az * 180 / np.pi
    az = np.where(az >= 180, az - 360, az)
    arr_sph = ejet.iono_sample[var][:, :, t]

    # fig = plt.figure(figsize=(8, 12))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    col = ax.contour(az, th, arr_sph / norm, vmin=vmin, vmax=vmax, cmap=cmap, alpha=1)
    cbar = plt.colorbar(col, ax=ax, fraction=0.0455, pad=0.04)
    cbar.ax.set_ylabel(name + unitlabel)

    extent = [az.min(), az.max(), th.min(), th.max()]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines("50m")

    ax.scatter(long0, lat0, color="r", s=40, edgecolors="w", zorder=2)
    txt = ax.text(long0 + 0.4, lat0, ejet.station, size=12, zorder=2, color="black")
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    ### Legacy code for high resolution local maps - may use in future
    # fname = os.path.join('/path/to/cartopy/data/raster/natural_earth/',
    # 'NE1_50M_SR_W.tif')
    # from matplotlib.image import imread
    # ax1.imshow(imread(fname), origin='upper', transform=ccrs.PlateCarree(),
    #         extent=[-180,180,-90,90])

    plt.xlabel(r"$Y$ / $km$")
    plt.ylabel(r"$X$ / $km$")

    if UT_time is not None:
        plt.title(UT_time)
    else:
        plt.title(r"$t$ = " + str(time))

    plt.tight_layout()

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()


def plot_electrojet(
    ejet, time, which=None, plotFAC=False, disp=True, filename=None, fileformat="png"
):
    """Plot the electrojet, ionospheric electric potential in the electrojet region.

    Args:
    ----
        ejet (eletproject): Elecproject object containing horizontal current data within
        a given region.
        time (int, optional): Simulation time in seconds to plot.
        which (string, optional): Specific current element to plot
        [None,'hall','pedersen']. Defaults to None.
        plotFAC (bool,optional): Flag to plot FACs as colourmap instead of electrojet
        magnitude. Defaults to False.
        disp (bool, optional): Choose whether to display the plot in case you just wish
        to save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    import matplotlib.patheffects as PathEffects

    # Get timestep
    t = np.where(ejet.times == time)[0][0]

    # Extract grid variables
    X, Y = ejet.X, ejet.Y
    if which == "hall":
        j_perp = ejet.arr["j_perp_hall"][:, :, :, t]
    elif which == "pedersen":
        j_perp = ejet.arr["j_perp_pedersen"][:, :, :, t]
    else:
        j_perp = ejet.arr["j_perp"][:, :, :, t]
    plotfac = ejet.arr["FAC"][:, :, t]

    # Plot
    # fig = plt.figure(figsize=(8, 12))
    ax = plt.subplot(111)
    # Identify station location
    ax.plot(0, 0, "ro", markersize=6, markeredgecolor="w")
    txt = ax.text(40, 0, ejet.station, size=12, zorder=2, color="black")
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    # Plot filled contour over region
    if plotFAC:
        col = ax.contourf(
            Y / 1e3,
            X / 1e3,
            plotfac,
            50,
            cmap="bwr",
            norm=colors.SymLogNorm(
                linthresh=1e-5, linscale=5e-1, base=10, vmin=-2e-6, vmax=2e-6
            ),
        )
    else:
        col = ax.contourf(
            Y / 1e3, X / 1e3, np.linalg.norm(j_perp, axis=0), 50, cmap="OrRd"
        )
    cbar = plt.colorbar(col, ax=ax, fraction=0.0455, pad=0.04)

    # Plot vectors at each grid point
    ax.quiver(Y / 1e3, X / 1e3, j_perp[1, :, :], j_perp[0, :, :], color="black")

    ax.set_xlim(-ejet.size / 1e3, ejet.size / 1e3)
    ax.set_ylim(-ejet.size / 1e3, ejet.size / 1e3)
    ax.set_aspect("equal")

    plt.xlabel(r"$Y$ / $km$")
    plt.ylabel(r"$X$ / $km$")
    if plotFAC:
        cbar.ax.set_ylabel(r"$j_{\|\|}$ / Am$^{-1}$")
    else:
        cbar.ax.set_ylabel(r"$j_\perp$ / Am$^{-1}$")

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight")

    if disp:
        plt.show()
    else:
        plt.close()
