"""Module for visualising ionospheric data."""

import datetime as dt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm


def plot_polar(
    iono,
    plt_list,
    hemisphere=None,
    th_max=40.0,
    t0_UT=None,
    show_time=True,
    pcolormesh=False,
    disp=True,
    filename=None,
    fileformat="png",
):
    """Plot a polar contour map of an ionospheric variable on one or both hemispheres.

    Takes plotting arguments from plt_list, with relevant examples given in the Jupyter
    notebook provided.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        plt_list (list): List containing variable name and dict of plotting arguments
        (see example notebook).
        hemisphere (str, optional): Choose specific hemisphere to plot if only one is
        desired, 'N' for North or 'S' for south. Defaults to None.
        th_max (float, optional): The lower bound of the plot in degrees colatitude,
        will change to (180-th_max) if hemi = 'S'. Defaults to 40.
        t0_UT (datetime, optional): UT datetime corresponding to t=0 in the simulation,
        for UT timestamping, if applicable. Defaults to None.
        show_time (bool, optional): Choose whether to show a timestamp.
        Defaults to True.
        pcolormesh (bool, optional): Choose whether to plot using pcolormesh rather than
        contourf. Defaults to False.
        disp (bool, optional): Choose whether to display the plot in case you just to
        save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    # Load plotting parameters
    var, params = plt_list

    # Variable name for labelling
    if "name" in params:
        name = params["name"]
    else:
        name = ""

    # If variable is a vector quantity, need to specify which component to plot
    if len(iono.arr[var].shape) > 2:
        if "comp" in params:
            if params["comp"] == "x":

                def comp_func(x):
                    return x[:, :, 0]
            elif params["comp"] == "y":

                def comp_func(x):
                    return x[:, :, 1]
            elif params["comp"] == "z":

                def comp_func(x):
                    return x[:, :, 2]
            elif params["comp"] == "mag":

                def comp_func(x):
                    return np.linalg.norm(x, axis=2)
            else:
                raise RuntimeError(
                    "Component ('x', 'y', 'z' or 'mag') must be specified via parameter"
                    " 'comp'" + "for vector quantity '" + var + "."
                )
        else:
            raise RuntimeError(
                "Component ('x', 'y', 'z' or 'mag') must be specified via parameter "
                "'comp'" + "for vector quantity '" + var + "."
            )
    else:

        def comp_func(x):
            return x

    # Unit of quantity for labelling
    if "unit" in params:
        unit = params["unit"]
        unitlabel = " / " + params["unit"]
    else:
        unit = ""
        unitlabel = ""

    # Normalisation based on chosen unit
    if "norm" in params:
        norm = params["norm"]
    else:
        norm = 1.0

    # Minimum value for contour bounds
    if "min" in params:
        vmin = params["min"]
    else:
        vmin = None

    # Maximum value for contour bounds
    if "max" in params:
        vmax = params["max"]
    else:
        vmax = None

    # Colour map for contour
    if "cmap" in params:
        cmap = params["cmap"]
    else:
        cmap = "RdBu_r"

    # Plot contour lines (or not)
    if "contours" in params:
        clines = params["contours"]
    else:
        clines = False

    # Load OCB coordinates if provided
    if "OCBs" in params:
        OCBs = params["OCBs"]
        show_OCB = True
    else:
        show_OCB = False

    plt.style.use("default")
    warnings.filterwarnings("ignore")  # , module = "matplotlib\..*" )

    # Calculate CPCP and TFAC for labelling
    if var == "FAC":
        from .analysis import calc_TFAC

        TFAC_N, TFAC_S = calc_TFAC(iono)
    if var == "phi":
        from .analysis import calc_CPCP

        CPCP_N, CPCP_S = calc_CPCP(iono)

    # Set up axes
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot2grid((1, 52), (0, 0), colspan=16, projection="polar")
    col1 = plt.subplot2grid((1, 52), (0, 18), colspan=1)
    ax2 = plt.subplot2grid((1, 52), (0, 28), colspan=16, projection="polar")
    col2 = plt.subplot2grid((1, 52), (0, 46), colspan=1)

    # Define which hemisphere(s) to plot
    if hemisphere is None:
        ax = (ax1, ax2)
        col = (col1, col2)
        hemispheres = ("N", "S")
    else:
        ax = [ax1]
        col = [col1]
        hemispheres = [hemisphere]
        plt.delaxes(ax2)
        plt.delaxes(col2)

    eq = len(iono.th) // 2

    # Start plotting
    for i in range(len(ax)):
        ax[i].set_theta_zero_location("S")
        ax[i].set_ylim(0, th_max)
        ax[i].tick_params(axis="y", labelsize=12)
        ax[i].set_yticks(np.arange(0, th_max + 1, 10).astype("int"))

        hemi = hemispheres[i]
        if hemi == "N":
            # Northern hemisphere
            az = iono.az_cyc
            daz = np.append(iono.daz_cyc, [iono.daz_cyc[0]])
            th = iono.th[:eq]  # N
            dth = iono.dth[:eq]
            colat, azim = np.meshgrid(th * 180 / np.pi, az)
            arr = comp_func(iono.arr_cyc[var])[:eq, :].T / norm  # N
            ax[i].set_xticklabels(
                ["00", "", "06", "", "12", "", "18", ""], fontsize=14
            )  # N
            if var == "FAC":
                polarcap_val = "TFAC = " + "%.1f" % (TFAC_N / 1e6) + " MA"
            elif var == "phi":
                polarcap_val = "CPCP = " + "%.1f" % (CPCP_N / 1e3) + " kV"
            hemi_label = "North"
        else:
            # Southern hemisphere
            az = iono.az_cyc
            daz = np.append(iono.daz_cyc, [iono.daz_cyc[0]])
            th = iono.th[eq:]  # S
            th = np.pi - th[::-1]  # S
            dth = iono.dth[eq::-1]
            colat, azim = np.meshgrid(th * 180 / np.pi, az)
            arr = (comp_func(iono.arr_cyc[var])[eq:, ::-1].T)[:, ::-1] / norm  # S
            ax[i].set_xticklabels(
                ["00", "", "18", "", "12", "", "06", ""], fontsize=14
            )  # S
            ax[i].set_yticklabels(
                [str(180 - i) for i in np.arange(0, th_max + 1, 10).astype("int")],
                fontsize=12,
            )  # S
            if var == "FAC":
                polarcap_val = "TFAC = " + "%.1f" % (TFAC_S / 1e6) + " MA"
            elif var == "phi":
                polarcap_val = "CPCP = " + "%.1f" % (CPCP_S / 1e3) + " kV"
            hemi_label = "South"

        # Specify colorbar limits
        con_extend = "neither"
        if vmax is None:
            vmax = np.max(arr)
        if vmin is None:
            vmin = np.min(arr)
        if vmin < 0 and vmax > 0:
            vmin, vmax = -np.max(np.abs([vmin, vmax])), np.max(np.abs([vmin, vmax]))
        if vmax < np.max(arr):
            con_extend = "max"
        if vmin > np.min(arr):
            con_extend = "both" if con_extend == "max" else "min"

        # Number of levels for contours
        N_lines = 11
        N_fills = 50
        if vmax <= 0 or vmin >= 0:
            N_lines = (N_lines + 1) / 2
            N_fills /= 2

        # Plot contour maps
        if vmax == vmin:
            pcolormesh = True
        if pcolormesh:
            pol = ax[i].pcolormesh(
                azim - np.meshgrid(dth / 2 * 180 / np.pi, daz / 2)[1],
                colat - np.meshgrid(dth / 2 * 180 / np.pi, daz / 2)[0],
                arr,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                zorder=0,
            )
            ax[i].grid(alpha=1, zorder=3)
            ax[i].set_facecolor("lightgray")
        else:
            line_levs = np.arange(vmin, vmax + vmax / N_lines, (vmax - vmin) / N_lines)
            fill_levs = np.arange(vmin, vmax + vmax / N_fills, (vmax - vmin) / N_fills)
            pol = ax[i].contourf(
                azim,
                colat,
                arr,
                fill_levs,
                extend=con_extend,
                cmap=cmap,
                zorder=0,
                alpha=0.95,
            )
            for c in pol.collections:
                c.set_rasterized(True)
        if clines:
            con = ax[i].contour(
                azim,
                colat,
                arr,
                line_levs,
                colors="black",
                linestyles="solid",
                linewidths=0.5,
            )
            for c in con.collections:
                c.set_rasterized(True)
        cbar = plt.colorbar(pol, cax=col[i], format="%.1f")
        cbar.ax.set_ylabel(name + unitlabel, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Add subplot labels
        ax[i].text(
            -3 * np.pi / 4 + 0.02,
            104 * th_max / 50,
            "Min = " + "%.1f" % np.min(arr) + " " + unit,
            family="sans-serif",
            ha="left",
            fontsize=14,
        )
        ax[i].text(
            -3 * np.pi / 4 + 0.065,
            100 * th_max / 50,
            "Max = " + "%.1f" % np.max(arr) + " " + unit,
            family="sans-serif",
            ha="left",
            fontsize=14,
        )
        if var in ["FAC", "phi"]:
            ax[i].text(
                -3 * np.pi / 4 + 0.110,
                96 * th_max / 50,
                polarcap_val,
                family="sans-serif",
                ha="left",
                fontsize=14,
            )
        ax[i].text(
            -3 * np.pi / 4 + 0.168,
            92 * th_max / 50,
            hemi_label,
            family="sans-serif",
            ha="left",
            fontsize=18,
        )
        if show_time:
            if t0_UT is not None:
                ax[i].text(
                    -np.pi / 4 - 0.05,
                    98 * th_max / 50,
                    (t0_UT + dt.timedelta(seconds=int(iono.time))).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    family="sans-serif",
                    ha="left",
                    fontsize=14,
                )
            else:
                ax[i].text(
                    -np.pi / 4 - 0.05,
                    98 * th_max / 50,
                    r"$t$ = " + str(iono.time) + " s",
                    family="sans-serif",
                    ha="left",
                    fontsize=14,
                )

        # Plot open-closed boundary
        if show_OCB:
            for OCB in OCBs:
                if hemi == "N" and OCB[0, 1] < np.pi / 2:
                    ax[i].plot(
                        OCB[:, 0],
                        OCB[:, 1] * 180 / np.pi,
                        10,
                        color="k",
                        linestyle="solid",
                        lw=1.5,
                        alpha=1,
                        rasterized=True,
                    )
                elif hemi == "S" and OCB[0, 1] > np.pi / 2:
                    ax[i].plot(
                        OCB[:, 0],
                        (np.pi - OCB[:, 1]) * 180 / np.pi,
                        color="k",
                        linestyle="solid",
                        lw=1.5,
                        alpha=1,
                        rasterized=True,
                    )

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat, bbox_inches="tight", transparent=True)

    if disp:
        plt.show()
    else:
        plt.close()


def lon_bound(x):
    """Wrap longitude values to -180 to 180 degrees."""
    return np.where(x > 180, x - 360, x)  # some hackery for pesky 0 longitude boundary


def plot_GEO(
    iono,
    plt_list,
    t0_UT,
    region="Global",
    hemisphere=None,
    show_time=True,
    show_stations=True,
    pcolormesh=False,
    disp=True,
    filename=None,
    fileformat="png",
):
    """Plot a polar contour map of an ionospheric variable on one or both hemispheres.

    Takes plotting arguments from plt_list, with relevant examples given in the Jupyter
    notebook provided.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        plt_list (list): List containing variable name and dict of plotting arguments
        (see example notebook).
        t0_UT (datetime): UT datetime corresponding to t=0 in the simulation, for UT
        timestamping, if applicable.
        region (list, optional): Geographic region to plot, from 'Global', 'Europe' or
        'UK' Defaults to 'Global'.
        hemisphere (str, optional): Choose specific hemisphere to plot for 'Global'
        region if desired, 'N' for North or 'S' for south. Defaults to None.
        show_time (bool, optional): Choose whether to show a timestamp.
        Defaults to True.
        show_stations (bool, optional): Choose whether to show a subset of regional
        geomagnetic station labels. Defaults to True.
        pcolormesh (bool, optional): Choose whether to plot using pcolormesh rather than
        contourf. Defaults to False.
        disp (bool, optional): Choose whether to display the plot in case you just to
        save the output. Defaults to True.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    # Load plotting parameters
    var, params = plt_list

    # Variable name for labelling
    if "name" in params:
        name = params["name"]
    else:
        name = ""

    # If variable is a vector quantity, need to specify which component to plot
    if len(iono.arr[var].shape) > 2:
        if "comp" in params:
            if params["comp"] == "x":

                def comp_func(x):
                    return x[:, :, 0]
            elif params["comp"] == "y":

                def comp_func(x):
                    return x[:, :, 1]
            elif params["comp"] == "z":

                def comp_func(x):
                    return x[:, :, 2]
            elif params["comp"] == "mag":

                def comp_func(x):
                    return np.linalg.norm(x, axis=2)
            else:
                raise RuntimeError(
                    "Component ('x', 'y', 'z' or 'mag') must be specified via parameter"
                    " 'comp'" + "for vector quantity '" + var + "."
                )
        else:
            raise RuntimeError(
                "Component ('x', 'y', 'z' or 'mag') must be specified via parameter "
                "'comp'" + "for vector quantity '" + var + "."
            )
    else:

        def comp_func(x):
            return x

    # Unit of quantity for labelling
    if "unit" in params:
        unit = params["unit"]
        unitlabel = " / " + params["unit"]
    else:
        unit = ""
        unitlabel = ""

    # Normalisation based on chosen unit
    if "norm" in params:
        norm = params["norm"]
    else:
        norm = 1.0

    # Minimum value for contour bounds
    if "min" in params:
        vmin = params["min"]
    else:
        vmin = None

    # Maximum value for contour bounds
    if "max" in params:
        vmax = params["max"]
    else:
        vmax = None

    # Colour map for contour
    if "cmap" in params:
        cmap = params["cmap"]
    else:
        cmap = "RdBu_r"

    # Plot contour lines (or not)
    if "contours" in params:
        clines = params["contours"]
    else:
        clines = False

    plt.style.use("default")

    # Define UT time and find Sun-Earth line for geographic plotting
    time_UT = t0_UT + dt.timedelta(seconds=int(iono.time))
    import cartopy.crs as ccrs

    from ..geomagnetic import coordinates as coords

    _, ss_az = coords.subsolar_angles(
        np.array([time_UT])
    )  # angle between MAG and SM X-axes

    # Specify geographic region and relevant plotting parameters
    if region == "Global":
        _, glon = coords.GEO_to_MAG(np.pi / 2, 0, np.array([time_UT]), inv=True)
        proj1 = ccrs.NearsidePerspective(
            central_longitude=(glon + ss_az) * 180 / np.pi + 180, central_latitude=90
        )
        _, glon = coords.GEO_to_MAG(-np.pi / 2, 0, np.array([time_UT]), inv=True)
        proj2 = ccrs.NearsidePerspective(
            central_longitude=(glon + ss_az) * 180 / np.pi + 180, central_latitude=-90
        )
        dlat, dlon = 1, 1  # lat/lon spacing for GEO grid
        lat_labels = [15, 30, 45, 60, 75]  # axis labels for lat
        lon_labels = [-135, -90, -45, 0, 45, 90, 135, 180]  # axis labels for lon
        coastres = "110m"  # coastline resolution
        ytxt_max, ytxt_min, ytxt_date = (
            1.09,
            1.15,
            -0.15,
        )  # DO NOT CHANGE - label coordinates
        stations_N = [
            "ABK",
            "DIK",
            "CCS",
            "BRW",
            "CMO",
            "FCC",
            "LRV",
            "NAQ",
            "PBK",
            "SNK",
            "TIK",
            "YKC",
            "BFE",
            "LER",
            "OTT",
            "MEA",
            "SIT",
        ]
        stations_S = ["EYR", "CNB", "VOS", "PST", "HER", "CSY", "VNA", "PIL", "LRM"]
        dstns = 2
    elif region == "UK":
        proj1 = ccrs.TransverseMercator(0, approx=True)
        dlat, dlon = 0.5, 0.5  # lat/lon spacing for GEO grid
        hemisphere = "N"  # always North, obviously
        latmin, latmax, lonmin, lonmax = (
            49.6,
            61,
            -10.5,
            2,
        )  # coordinate range of region
        maplatmin, maplatmax, maplonmin, maplonmax = (
            49,
            61,
            -13,
            2.5,
        )  # lat/lon range appearing on map - for colormap scaling
        lat_labels = [50, 52, 54, 56, 58, 60]  # axis labels for lat
        lon_labels = [-12, -10, -8, -6, -4, -2, 0, 2]  # axis labels for lon
        coastres = "10m"  # coastline resolution
        ytxt_max, ytxt_min, ytxt_date = (
            1.02,
            1.07,
            -0.16,
        )  # DO NOT CHANGE - label coordinates
        stations = ["LER", "ESK", "HAD", "VAL"]
        dstns = 0.25
    elif region == "Europe":
        proj1 = ccrs.TransverseMercator(30, approx=True)
        dlat, dlon = 1, 1  # lat/lon spacing for GEO grid
        hemisphere = "N"  # always North, obviously
        latmin, latmax, lonmin, lonmax = 34, 69.5, -8, 50  # coordinate range of region
        maplatmin, maplatmax, maplonmin, maplonmax = (
            30,
            73.5,
            -30.5,
            70,
        )  # lat/lon range appearing on map - for colormap scaling
        lat_labels = np.arange(30, 71, 10)  # axis labels for lat
        lon_labels = np.arange(-60, 71, 10)  # axis labels for lon
        coastres = "50m"  # coastline resolution
        ytxt_max, ytxt_min, ytxt_date = (
            1.05,
            1.10,
            -0.2,
        )  # DO NOT CHANGE - label coordinates
        stations = [
            "ABK",
            "LRV",
            "UPS",
            "BFE",
            "NGK",
            "LER",
            "ESK",
            "HAD",
            "IZN",
            "DUR",
            "EBR",
            "BOX",
            "KIV",
            "CLF",
        ]
        dstns = 1

    # Generate plots with specified projection
    if region == "Global":
        if hemisphere is None:
            fig = plt.figure(figsize=(15, 6))
            ax1 = plt.subplot2grid((1, 52), (0, 0), colspan=16, projection=proj1)
            col1 = plt.subplot2grid((1, 52), (0, 19), colspan=1)
            ax2 = plt.subplot2grid((1, 52), (0, 28), colspan=16, projection=proj2)
            col2 = plt.subplot2grid((1, 52), (0, 46), colspan=1)
            hemispheres = ["N", "S"]
            axs = [ax1, ax2]
            col = [col1, col2]
        else:
            fig = plt.figure(figsize=(7.5, 6))
            if hemisphere == "N":
                ax1 = plt.subplot2grid((1, 26), (0, 0), colspan=16, projection=proj1)
            if hemisphere == "S":
                ax1 = plt.subplot2grid((1, 26), (0, 0), colspan=16, projection=proj2)
            col1 = plt.subplot2grid((1, 26), (0, 18), colspan=1)
            hemispheres = hemisphere
            axs = [ax1]
            col = [col1]
    elif region == "Europe" or region == "UK":
        if region == "Europe":
            fig = plt.figure(figsize=(7.75, 6))
            ax1 = plt.subplot2grid((1, 36), (0, 0), colspan=26, projection=proj1)
            col1 = plt.subplot2grid((1, 36), (0, 28), colspan=1)
        if region == "UK":
            fig = plt.figure(figsize=(6.5, 7))
            ax1 = plt.subplot2grid((1, 24), (0, 0), colspan=23, projection=proj1)
            col1 = plt.subplot2grid((1, 24), (0, 23), colspan=1)
        hemispheres = hemisphere
        axs = [ax1]
        col = [col1]

    # Specify fiddly Cartopy plotting parameters
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    for i, ax in enumerate(axs):
        ax.coastlines(coastres, color="grey")
        if region != "Global":
            ax.set_extent((lonmin, lonmax, latmin, latmax), ccrs.PlateCarree())
        else:
            ax.set_global()

            def lon_bound(x):
                return x

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlocator, gl.ylocator = (
            mticker.FixedLocator(lon_labels),
            mticker.FixedLocator([1, -1][i] * np.array(lat_labels)),
        )
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.right_labels, gl.top_labels = False, False

    eq = len(iono.th) // 2

    for i in range(len(hemispheres)):
        hemi = hemispheres[i]
        az = (iono.az + np.pi + ss_az) % (2 * np.pi) * 180 / np.pi  # in MAG coordinates
        if hemi == "N":
            # Northern hemisphere
            th = iono.th[:eq] * 180 / np.pi  # N
            colat, azim = np.meshgrid(th, az)
            arr = comp_func(iono.arr[var])[:eq, :].T / norm  # N
        else:
            # Southern hemisphere
            th = iono.th[eq:] * 180 / np.pi  # S
            colat, azim = np.meshgrid(th, az)
            arr = comp_func(iono.arr[var])[eq:, :].T / norm  # S

        # Define geomagnetic coordinates
        lat_MAG, lon_MAG = coords.GEO_to_MAG(
            np.pi / 2 - colat * np.pi / 180,
            azim * np.pi / 180,
            np.array([time_UT]),
            inv=True,
        )
        lat_MAG, lon_MAG = lat_MAG * 180 / np.pi, lon_MAG * 180 / np.pi

        # Do tedious geographic coordinate transformation,
        # add periodicity at 0 longitude
        lats, lons = np.arange(0, 90 + dlat, dlat), np.arange(0, 360, dlon)
        if hemi == "S":
            lats = -lats[::-1]
        lat, lon = np.meshgrid(lats, lons, indexing="ij")
        from scipy import interpolate as interp

        zero_flag = np.any(arr < 0)  # is floor > zero?...
        arr = interp.griddata(
            np.array([lat_MAG.ravel(), lon_bound(lon_MAG.ravel())]).T,
            arr.ravel(),
            (lat, lon_bound(lon)),
            method="linear",
            fill_value=0,
        )
        if not zero_flag:  # ... if so ...
            arr[
                arr < 0
            ] = 0  # ... reinforce this, in case interpolation creates artificial
            # negative values
        from cartopy.util import add_cyclic_point

        arr, lons = add_cyclic_point(
            arr, coord=lons
        )  # cartopy hackery to handle pesky longitude boundary
        lat, lon = np.meshgrid(lats, lons, indexing="ij")
        lons = np.where(lons > 180, lons - 360, lons)

        if not region == "Global":
            arr_loc, lons_loc = arr[:, lons > maplonmin - 1], lons[lons > maplonmin - 1]
            arr_loc, lons_loc = (
                arr_loc[:, lons_loc < maplonmax + 1],
                lons_loc[lons_loc < maplonmax + 1],
            )
            arr_loc, lats_loc = (
                arr_loc[lats > maplatmin - 1, :],
                lats[lats > maplatmin - 1],
            )
            arr_loc, lats_loc = (
                arr_loc[lats_loc < maplatmax + 1, :],
                lats_loc[lats_loc < maplatmax + 1],
            )
        else:
            arr_loc = arr

        # Plot pole to conceal ugly singularity distortion
        if region == "Global":
            axs[i].scatter(
                0,
                (90 if hemi == "N" else -90),
                s=10,
                color="grey",
                transform=ccrs.PlateCarree(),
            )
            stations = stations_N if hemi == "N" else stations_S

        # Plot regional stations
        def get_stations(names):
            from gorgon_tools.geomagnetic.coordinates import get_station_coords

            x, y = [], []
            for n in names:
                lat, lon = get_station_coords(n)
                x.append(90 - lat * 180 / np.pi)
                if lon > np.pi:
                    y.append(lon * 180 / np.pi - 360)
                else:
                    y.append(lon * 180 / np.pi)

            return np.array(x), np.array(y)

        if show_stations:
            import matplotlib.patheffects as PathEffects

            st_lats, st_lons = get_stations(stations)
            axs[i].scatter(
                st_lons,
                st_lats,
                s=15,
                zorder=10,
                color="limegreen" if cmap != "Greens" else "red",
                edgecolor="k",
                transform=ccrs.PlateCarree(),
            )
            for istn, s in enumerate(stations):
                txt = axs[i].text(
                    st_lons[istn] - dstns / 2,
                    st_lats[istn] - dstns,
                    s,
                    color="darkgreen" if cmap != "Greens" else "darkred",
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    horizontalalignment="right",
                    zorder=11,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="w")]
                )

        # Specify colorbar limits
        con_extend = "neither"
        if vmax is None:
            vmax = np.max((np.max(arr_loc), 0.1))
        if vmin is None:
            vmin = np.min((np.min(arr_loc), -0.1))
        if vmin < 0 and vmax > 0:
            vmin, vmax = -np.max(np.abs([vmin, vmax])), np.max(np.abs([vmin, vmax]))
        if vmax < np.max(arr_loc):
            con_extend = "max"
        if vmin > np.min(arr_loc):
            con_extend = "both" if con_extend == "max" else "min"

        # Define number of contour levels
        N_lines = 11
        N_fills = 50
        if vmax <= 0 or vmin >= 0:
            N_lines = (N_lines + 1) / 2
            N_fills /= 2

        # Plot contour map
        if vmax == vmin:
            pcolormesh = True
        if pcolormesh:
            pol = axs[i].pcolormesh(
                lon - dlon / 2,
                lat - dlat / 2,
                arr,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                zorder=0,
                transform=ccrs.PlateCarree(),
            )
            axs[i].grid(b=True, alpha=1, zorder=3)
            axs[i].set_facecolor("lightgray")
        else:
            line_levs = np.arange(vmin, vmax + vmax / N_lines, (vmax - vmin) / N_lines)
            fill_levs = np.arange(vmin, vmax + vmax / N_fills, (vmax - vmin) / N_fills)
            pol = axs[i].contourf(
                lon,
                lat,
                arr,
                fill_levs,
                extend=con_extend,
                cmap=cmap,
                zorder=0,
                alpha=0.95,
                transform=ccrs.PlateCarree(),
            )
            for c in pol.collections:
                c.set_rasterized(True)
        if clines:
            con = axs[i].contour(
                lon,
                lat,
                arr,
                line_levs,
                colors="black",
                linestyles="solid",
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )
            for c in con.collections:
                c.set_rasterized(True)
        cbar = plt.colorbar(pol, cax=col[i], format="%.2f")
        cbar.ax.set_ylabel(name + unitlabel, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Add labelling
        axs[i].text(
            0,
            ytxt_min,
            "Min = " + "%.2f" % np.min(arr_loc) + " " + unit,
            family="sans-serif",
            ha="left",
            fontsize=14,
            transform=axs[i].transAxes,
        )
        axs[i].text(
            0,
            ytxt_max,
            "Max = " + "%.2f" % np.max(arr_loc) + " " + unit,
            family="sans-serif",
            ha="left",
            fontsize=14,
            transform=axs[i].transAxes,
        )
        if show_time:
            axs[i].text(
                0,
                ytxt_date,
                time_UT.strftime("%Y-%m-%d %H:%M"),
                family="sans-serif",
                ha="left",
                fontsize=14,
                transform=axs[i].transAxes,
            )
        vmin, vmax = None, None

    # Save image
    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(
            filename, format=fileformat, bbox_inches="tight", transparent=True, fig=fig
        )

    if disp:
        plt.show()
    else:
        plt.close()


def plot_keogram(
    iono,
    starttime=None,
    endtime=None,
    t0_UT=None,
    maxFAC=None,
    filename=None,
    fileformat="png",
):
    """Plot a keogram across the dawn-dusk meridian.

    Plots a keogram (FAC intensity [colours] at different latitudes [y-axis] over time
    [x-axis]) across the dawn-dusk meridian.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        starttime (int, optional): Initial time in seconds to plot, if None will use
        first available timestep. Defaults to None.
        endtime (int, optional): Final time in seconds to plot, if None will use last
        available timestep. Defaults to None.
        t0_UT (datetime, optional): Date and time in UT corresponding to t=0, if a UT
        time axis is desired. Defaults to None.
        maxFAC (float, optional): Peak FAC value for colormap, if None use maximum value
        over timerange. Defaults to None.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    sns.set_style("ticks")
    import matplotlib.dates as mdates
    from matplotlib import ticker

    t_init = iono.timestep(iono.time)  # store current time to revert to later
    n_lat = int(
        (40 * np.pi / 180) / iono.dth[0]
    )  # will plot over a 50deg latitude range using the highest resolution
    eq = len(iono.th)  # store index of equator
    if starttime is None:
        starttime = iono.times[0]
    if endtime is None:
        endtime = iono.times[-1]
    times = range(iono.timestep(starttime), iono.timestep(endtime) + 1)
    FAC = np.zeros([eq * 2, len(times)])

    # Construct an array of thetas over whole meridian
    thtot = np.zeros([eq * 2])
    thtot[:eq] = iono.th
    thtot[eq : 2 * eq] = iono.th + np.pi
    dawn = np.argmin(abs(iono.az - np.pi / 2))
    dusk = np.argmin(abs(iono.az - 3 * np.pi / 2))
    for t in range(len(times)):
        iono.import_timestep(times[t])

        # Obtain FAC at time t and shift array such that north is the first half,
        # south is second
        FACth = np.zeros([eq * 2])
        FACth[eq : 2 * eq] = np.flip(iono.arr["FAC"][:, dusk], 0)
        FACth[:eq] = iono.arr["FAC"][:, dawn]
        FACthShift = 0 * FACth
        for i in range(eq * 2):
            if thtot[i] < 3 * np.pi / 2:
                FACthShift[i + int(eq / 2)] = FACth[i] * 1e6
            else:
                FACthShift[i - 3 * int(eq / 2)] = FACth[i] * 1e6
        FAC[:, t] = FACthShift

    # Now plot
    _, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    cmap = plt.get_cmap("RdBu_r")
    FAC_N, FAC_S = (
        FAC[n_lat : eq - n_lat + 1, :],
        FAC[eq + n_lat : 2 * eq - n_lat + 1, :],
    )
    if maxFAC is None:
        N_maxFAC = np.max(FAC_N)
        S_maxFAC = np.max(FAC_S)
    else:
        N_maxFAC, S_maxFAC = maxFAC, maxFAC
    N_fill_levs = np.arange(-N_maxFAC, N_maxFAC * (1 + 1 / 50), N_maxFAC / 50)
    S_fill_levs = np.arange(-S_maxFAC, S_maxFAC * (1 + 1 / 50), S_maxFAC / 50)
    N_norm = BoundaryNorm(N_fill_levs, ncolors=cmap.N, clip=True)
    S_norm = BoundaryNorm(S_fill_levs, ncolors=cmap.N, clip=True)
    if t0_UT is None:
        time = [float(i) / 3600 for i in iono.times[times]]
    else:
        time = [t0_UT + dt.timedelta(seconds=int(t)) for i in iono.times[times]]
    time, th = np.meshgrid(
        time, thtot * 180 / np.pi - 90
    )  # meshgrid needed for pcolormesh
    col1 = ax[0].pcolormesh(
        time[n_lat : eq - n_lat + 1, :],
        th[n_lat : eq - n_lat + 1, :],
        FAC_N,
        norm=N_norm,
        cmap="RdBu_r",
        zorder=0,
    )
    ax[0].set_ylim(th[n_lat, 0], th[eq - n_lat, 0])
    ax[0].set_ylabel(r"North Colat. ($^\circ$)", fontsize=13)
    ax[0].text(time[0, 0] + 0.01 * (time[0, -1] - time[0, 0]), 38, "Dawn", fontsize=12)
    ax[0].text(time[0, 0] + 0.01 * (time[0, -1] - time[0, 0]), -45, "Dusk", fontsize=12)
    ax[0].set_yticklabels(["", "40", "20", "0", "20", "40"])
    cbar1 = plt.colorbar(col1, ax=ax[0], extend="both", format="%.1f")
    cbar1.ax.set_ylabel(r"$j_{||}$ ($\mu$A/m$^2$)", fontsize=13)

    col2 = ax[1].pcolormesh(
        time[eq + n_lat : 2 * eq - n_lat + 1, :],
        th[eq + n_lat : 2 * eq - n_lat + 1, :],
        FAC_S,
        norm=S_norm,
        cmap="RdBu_r",
        zorder=0,
        shading="auto",
    )
    ax[1].set_ylim(th[2 * eq - n_lat, 0], th[eq + n_lat, 0])
    ax[1].set_ylabel(r"South Colat. ($^\circ$)", fontsize=13)
    ax[1].text(time[0, 0] + 0.01 * (time[0, -1] - time[0, 0]), 142, "Dawn", fontsize=12)
    ax[1].text(time[0, 0] + 0.01 * (time[0, -1] - time[0, 0]), 225, "Dusk", fontsize=12)
    ax[1].set_yticklabels(["", "140", "160", "180", "160", "140"])
    cbar2 = plt.colorbar(col2, ax=ax[1], extend="both", format="%.1f")
    cbar2.ax.set_ylabel(r"$j_{||}$ ($\mu$A/m$^2$)", fontsize=13)
    if t0_UT is None:
        ax[1].set_xlabel(r"$t$ (h)", fontsize=14)
    else:
        ax[1].xaxis.set_major_locator(ticker.LinearLocator(20))
        locator = mdates.AutoDateLocator(minticks=6, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        ax[1].xaxis.set_major_locator(locator)
        ax[1].xaxis.set_major_formatter(formatter)

    ax[0].grid(alpha=0.9, zorder=3)
    ax[1].grid(alpha=0.9, zorder=3)

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat)

    plt.show()

    iono.import_timestep(t_init)


def plot_CPCP_TFAC_vs_time(
    iono, starttime=None, endtime=None, t0_UT=None, filename=None, fileformat="png"
):
    """Plot the (CPCP) and (TFAC) on the ionosphere over time.

    Plots the cross-polar cap potential (CPCP) and total (upward) field-aligned current
    (TFAC) on the ionosphere over time.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        starttime (int, optional): Initial time in seconds to plot, if None will use
        first available timestep. Defaults to None.
        endtime (int, optional): Final time in seconds to plot, if None will use last
        available timestep. Defaults to None.
        t0_UT (datetime, optional): Date and time in UT corresponding to t=0, if a UT
        time axis is desired. Defaults to None.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        fileformat (str, optional): File type to use if you wish to save the output.
        Defaults to 'png'.

    """
    sns.set_style("whitegrid")

    t_init = iono.timestep(iono.time)  # store current time to revert to later
    TFACs_N = []
    TFACs_S = []
    CPCPs_N = []
    CPCPs_S = []
    if starttime is None:
        starttime = iono.times[0]
    if endtime is None:
        endtime = iono.times[-1]
    times = range(iono.timestep(starttime), iono.timestep(endtime) + 1)

    from .analysis import calc_CPCP, calc_TFAC

    for t in times:
        iono.import_timestep(t)
        TFAC_N, TFAC_S = calc_TFAC(iono)
        CPCP_N, CPCP_S = calc_CPCP(iono)
        TFACs_N.append(TFAC_N / 1e6)
        TFACs_S.append(TFAC_S / 1e6)
        CPCPs_N.append(CPCP_N / 1e3)
        CPCPs_S.append(CPCP_S / 1e3)

    if t0_UT is None:
        time = [float(i) / 3600 for i in iono.times[times]]
    else:
        time = [t0_UT + dt.timedelta(seconds=int(t)) for i in iono.times[times]]

    _, ax1 = plt.subplots(figsize=(12, 6.25))

    ax1.plot(time, CPCPs_N, color="royalblue")
    ax1.plot(time, CPCPs_S, color="royalblue", linestyle="dashed")
    ax1.set_ylabel("CPCP / kV", color="royalblue", fontsize=14)
    ax1.tick_params("y", colors="royalblue", labelsize=12)
    ax1.tick_params("x", labelsize=13)
    ax1.set_ylim(0, max(np.max(CPCPs_N), np.max(CPCPs_S)) * 1.1)
    ax1.set_xlim(time[0], time[-1])

    ax2 = ax1.twinx()
    ax2.plot(time, TFACs_N, "firebrick")
    ax2.plot(time, TFACs_S, "firebrick", linestyle="dashed")
    ax2.set_ylabel("TFAC / MA", color="firebrick", fontsize=14)
    ax2.tick_params("y", colors="firebrick", labelsize=12)
    ax2.set_ylim(0, max(np.max(TFACs_N), np.max(TFACs_S)) * 1.1)
    if t0_UT is None:
        ax1.set_xlabel(r"$t$ (h)", fontsize=14)

    nticks = 6
    ax1.yaxis.set_major_locator(ticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(ticker.LinearLocator(nticks))
    # ax1.format_xdata = mdates.DateFormatter('%H:%M:%S')

    ax1.plot(0, 0, color="dimgray", label="North")
    ax1.plot(0, 0, color="dimgray", linestyle="dashed", label="South")

    ax1.legend(loc=2, fontsize=14, frameon=0, framealpha=1)

    if filename is not None:
        if filename[-3:] != fileformat:
            filename += "." + fileformat
        plt.savefig(filename, format=fileformat)
    plt.show()

    iono.import_timestep(t_init)
