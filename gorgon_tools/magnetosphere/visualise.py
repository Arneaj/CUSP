"""Module for visualising magnetospheric data."""
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from matplotlib.ticker import MultipleLocator

from ..geomagnetic.coordinates import SM_to_SMD, rot_y


def stretched_to_uniform(sim, arrs=None, d0=None):
    """Interpolate cell-centered arrays from a stretched grid to a uniform grid.

    Args:
    ----
        sim (Simulation): The simulation object containing the stretched grid
        and arrays.
        arrs (list, optional): A list of array names to interpolate. If None, all arrays
        are interpolated. Defaults to None.
        d0 (float, optional): The grid spacing of the uniform grid.
        If None, the minimum grid spacing of the stretched grid is used.
        Defaults to None.

    Returns:
    -------
        None

    """
    # Cell-centred arrays only
    if d0 is None:
        d0 = min(np.min(sim.dx), np.min(sim.dy), np.min(sim.dz))
    xstr, ystr, zstr = sim.xc, sim.yc, sim.zc
    sim.xb, sim.yb, sim.zb = (
        np.arange(max(-30, sim.xb[0]), min(60, sim.xb[-1]) + d0 / 2, d0),
        np.arange(max(-40, sim.yb[0]), min(40, sim.yb[-1]) + d0 / 2, d0),
        np.arange(max(-40, sim.zb[0]), min(40, sim.zb[-1]) + d0 / 2, d0),
    )
    sim.dx, sim.dy, sim.dz = (
        sim.xb[1:] - sim.xb[:-1],
        sim.yb[1:] - sim.yb[:-1],
        sim.zb[1:] - sim.zb[:-1],
    )
    sim.xc, sim.yc, sim.zc = (
        sim.xb[:-1] + sim.dx / 2,
        sim.yb[:-1] + sim.dy / 2,
        sim.zb[:-1] + sim.dz / 2,
    )
    sim.center = -np.array([sim.xb[0], sim.yb[0], sim.zb[0]])

    if arrs is None:
        arrs = sim.arr_names

    x, y, z = np.meshgrid(sim.xc, sim.yc, sim.zc, indexing="ij")
    xs, ys, zs = x.ravel(), y.ravel(), z.ravel()
    for var in arrs:
        var_int = interp.RegularGridInterpolator(
            (xstr, ystr, zstr), sim.arr[var], bounds_error=False, fill_value=None
        )
        if len(sim.arr[var].shape) > 3:
            sim.arr[var] = var_int(np.stack([xs, ys, zs]).T).reshape(
                [len(sim.xc), len(sim.yc), len(sim.zc), 3]
            )
        else:
            sim.arr[var] = var_int(np.stack([xs, ys, zs]).T).reshape(
                [len(sim.xc), len(sim.yc), len(sim.zc)]
            )


def stretched_to_uniform_2D(sim, arrs=None, d0=None):
    """Interpolate cell-centered arrays from a stretched grid to a uniform grid.

    Args:
    ----
        sim: A simulation object containing the stretched grid and cell-centered arrays.
        arrs: A list of names of the arrays to be interpolated. If None, all arrays will
        be interpolated.
        d0: The grid spacing of the uniform grid. If None, the minimum grid spacing of
        the stretched grid is used.

    Returns:
    -------
        None

    """
    # Cell-centred arrays only
    if d0 is None:
        d0 = min(np.min(sim.dx), np.min(sim.dz))
    xstr, zstr = sim.xc, sim.zc
    sim.xb, sim.zb = (
        np.arange(max(-30, sim.xb[0]), min(60, sim.xb[-1]) + d0, d0),
        np.arange(max(-40, sim.zb[0]), min(40, sim.zb[-1]) + d0, d0),
    )
    sim.dx, sim.dz = sim.xb[1:] - sim.xb[:-1], sim.zb[1:] - sim.zb[:-1]
    sim.xc, sim.zc = sim.xb[:-1] + sim.dx / 2, sim.zb[:-1] + sim.dz / 2
    sim.center = -np.array([sim.xb[0], sim.yb[0], sim.zb[0]])

    if arrs is None:
        arrs = sim.arr_names

    x, z = np.meshgrid(sim.xc, sim.zc, indexing="ij")
    xs, zs = x.ravel(), z.ravel()
    for var in arrs:
        sim.arr[var] = np.squeeze(sim.arr[var])
        var_int = interp.RegularGridInterpolator(
            (xstr, zstr), sim.arr[var], bounds_error=False
        )
        if len(sim.arr[var].shape) > 2:
            sim.arr[var] = var_int(np.stack([xs, zs]).T).reshape(
                [len(sim.xc), len(sim.zc), 3]
            )
        else:
            sim.arr[var] = var_int(np.stack([xs, zs]).T).reshape(
                [len(sim.xc), len(sim.zc)]
            )


def plot_slices(
    sim,
    plt_list,
    plt_coords=None,
    r_IB=4,
    mu=0,
    plot_flines=True,
    t_UT=None,
    disp=True,
    filename=None,
    fileformat="png",
):
    """Plot slices of a simulation variable.

    Args:
    ----
        sim (Simulation): The simulation object.
        plt_list (list): A list of tuples containing the variable name and plotting
        parameters.
        plt_coords (str, optional): The coordinate system to plot in. Defaults to None.
        r_IB (float, optional): The radius of the inner boundary. Defaults to 4.
        mu (float, optional): The tilt angle of the dipole. Defaults to 0.
        plot_flines (bool, optional): Whether to plot field lines. Defaults to True.
        t_UT (datetime.datetime, optional): The time in UT to plot the field lines.
        If None, the current simulation time is used. Defaults to None.
        disp (bool, optional): Whether to display the plot. Defaults to True.
        filename (str, optional): The name of the output file. If None, the plot is not
        saved. Defaults to None.
        fileformat (str, optional): The format of the output file. Defaults to "png".

    Returns:
    -------
        None

    """
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

    if "n_levs" in params:
        n_levs = params["n_levs"]
    else:
        n_levs = 100

    if "cmap" in params:
        cmap = params["cmap"]
    else:
        cmap = "RdBu_r"

    if "log" in params:
        log = params["log"]

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5))

    if plt_coords == "GSM":
        var = var + "_GSM"
        x, y, z = sim.xc_GSM, sim.yc_GSM, sim.zc_GSM
        xyz_sgn = np.array([-1, -1, 1])
    else:
        x, y, z = sim.xc, sim.yc, sim.zc
        xyz_sgn = np.array([1, 1, 1])

    iy = len(y) // 2
    arr_xz = sim.arr[var][:, iy, :] / norm
    z_xz, x_xz = np.meshgrid(z, x)
    z_xz, x_xz = z_xz.T, x_xz.T
    y_xz = z[iy]

    iz = len(z) // 2
    arr_xy = sim.arr[var][:, :, iz] / norm
    y_xy, x_xy = np.meshgrid(y, x)
    y_xy, x_xy = y_xy.T, x_xy.T

    r = np.sqrt(x_xz**2 + y_xz**2 + z_xz**2)
    mask_xz = r <= r_IB
    coords = ([x_xz, z_xz], [x_xy, y_xy])

    if plt_coords == "GSM":
        strm_vars = ("Bvec_c_GSM", "Bvec_c_GSM")
        strm_center = (
            np.array([len(sim.xb_GSM), len(sim.yb_GSM), 2 * sim.center_GSM[2]])
            - sim.center_GSM
        )
    else:
        strm_vars = ("Bvec_c", "Bvec_c")
        strm_center = sim.center

    strm_cols = np.array(("k", "dimgrey", "w"))

    for i, v in enumerate((arr_xz.T, arr_xy.T)):
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
                min_exp, max_exp, (max_exp - min_exp) / n_levs
            )  # np.power(10, levs_exp)
            v_masked = np.where(v < 10.0**min_exp, min_drawn_value, v)
            v_masked = np.where(v_masked > 10.0**max_exp, max_drawn_value, v_masked)
        else:
            levs = np.arange(1.05 * vmin, 1.05 * vmax, 1.05 * (vmax - vmin) / n_levs)
            v_masked = 1 * v

        v_masked[mask_xz] = vmin / 10

        if log:
            p = ax[i].contourf(
                coords[i][0],
                coords[i][1],
                v_masked,
                levs,
                norm=LogNorm(vmin=min_drawn_value, vmax=max_drawn_value),
                extend="both",
                cmap=cmap,
            )
        else:
            p = ax[i].contourf(
                coords[i][0], coords[i][1], v_masked, levs, extend="both", cmap=cmap
            )

        if t_UT is None:
            txt = ax[i].text(
                -20.0 * xyz_sgn[0],
                26.0 * xyz_sgn[2 - i],
                f"t = {sim.time} s",
                fontsize=15,
                color="k",
                fontfamily="monospace",
            )
            del_mu = 0
        else:
            txt = ax[i].text(
                -20.0 * xyz_sgn[0],
                26.0 * xyz_sgn[2 - i],
                t_UT.strftime("%Y-%m-%d %H:%M"),
                fontsize=15,
                color="k",
                fontfamily="monospace",
            )
            t0_UT = t_UT - dt.timedelta(seconds=int(sim.time))
            _, _, _, mu, del_mu = SM_to_SMD(
                0,
                0,
                1,
                np.array([t0_UT + dt.timedelta(seconds=int(t)) for t in sim.times]),
            )
            mu = mu * np.pi / 180  # convert to radians for appropriate use
            del_mu = del_mu[sim.timestep(sim.time)]
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="w")])
        d_arrow = -0.98 * r_IB * np.cos(del_mu), 0, 0.98 * r_IB * np.sin(del_mu)
        ax[i].arrow(
            0,
            0,
            d_arrow[i],
            d_arrow[2 - i],
            color="w",
            length_includes_head=True,
            head_width=1,
            head_length=1,
            zorder=2,
        )
        d_arrow = -0.98 * r_IB * np.sin(mu), 0, 0.98 * r_IB * np.cos(mu)
        ax[i].arrow(
            0,
            0,
            d_arrow[i],
            d_arrow[2 - i],
            color="b",
            length_includes_head=True,
            head_width=0.5,
            head_length=0.5,
            lw=1.5,
            zorder=2,
        )
        d_arrow = 0.98 * r_IB * np.sin(mu), 0, -0.98 * r_IB * np.cos(mu)
        ax[i].arrow(
            0,
            0,
            d_arrow[i],
            d_arrow[2 - i],
            color="r",
            length_includes_head=True,
            head_width=0.5,
            head_length=0.5,
            lw=1.5,
            zorder=2,
        )

        if plot_flines:
            from ..magnetosphere.fieldlines import fieldlines

            xs, seed_labels = gen_seeds(["xz", "xy"][i], r_IB, xyz_sgn[i], mu, sim)

            streams = fieldlines(10000, 0.1 * np.mean(sim.dx))
            if plt_coords == "GSM":
                strm_arr = sim.arr[strm_vars[i]][
                    ::-1,
                    ::-1,
                ].copy()
            else:
                strm_arr = sim.arr[strm_vars[i]].copy()
            if i == 0:
                strm_arr[:, :, :, i + 1] = 0
            streams.calc(
                xs,
                strm_arr,
                np.ones([3]) * np.mean(sim.dx),
                strm_center - 0.5 * np.mean(sim.dx),
                v_name=strm_vars[i],
            )
            streams.calc_linkage(sim)
            if i == 0:
                xmp = 0
                for s, pts in enumerate(streams.xs):
                    if s < len(streams.cell_data["link"]):
                        if (
                            streams.cell_data["link"][s] >= 5
                            and np.min(pts[:, 0]) < xmp
                        ):
                            xmp = np.min(pts[:, 0])
                for s, pts in enumerate(streams.xs):
                    if s < len(streams.cell_data["link"]):
                        col = strm_cols[
                            streams.cell_data["link"][s] == np.array([1, 5, 8])
                        ][0]
                        if not (seed_labels[s] == 2 and np.min(pts[:, 0]) > xmp):
                            if not (
                                streams.cell_data["link"][s] == 8 and seed_labels[s] > 0
                            ):
                                ax[i].plot(
                                    pts[:, 0], pts[:, 2], col, zorder=1, lw=1.5, alpha=1
                                )
            else:
                for s, pts in enumerate(streams.xs):
                    if s < len(streams.cell_data["link"]):
                        col = strm_cols[
                            streams.cell_data["link"][s] == np.array([1, 5, 8])
                        ][0]
                        ax[i].plot(pts[:, 0], pts[:, 1], col, zorder=1, lw=1.5, alpha=1)

        ax[i].set_facecolor("k")
        for c in p.collections:
            c.set_edgecolor("face")

        ax[i].tick_params(axis="both", labelsize=9)
        ax[i].set_xlim(
            max(-22 * xyz_sgn[0], sim.xb[0] * xyz_sgn[0]),
            min(40 * xyz_sgn[0], sim.xb[-1] * xyz_sgn[0]),
        )
        if i == 0:
            ax[i].set_ylim(
                max(-30 * xyz_sgn[2 - i], sim.yb[0] * xyz_sgn[2 - i]),
                min(30 * xyz_sgn[2 - i], sim.yb[-1] * xyz_sgn[2 - i]),
            )
        if i == 1:
            ax[i].set_ylim(
                max(-30 * xyz_sgn[2 - i], sim.zb[0] * xyz_sgn[2 - i]),
                min(30 * xyz_sgn[2 - i], sim.zb[-1] * xyz_sgn[2 - i]),
            )

        ax[i].xaxis.set_major_locator(MultipleLocator(10))
        ax[i].xaxis.set_minor_locator(MultipleLocator(2))
        ax[i].yaxis.set_major_locator(MultipleLocator(10))
        ax[i].yaxis.set_minor_locator(MultipleLocator(2))
        ax[i].set_rasterized(True)

        IB_mask(r_IB, ax=ax[i], color="k", zangle=90 - del_mu * 180 / np.pi * (1 - i))

    if plt_coords == "GSM":
        ax[0].set_ylabel(r"$Z_{GSM}$ / $R_E$", fontsize=15, labelpad=-2)
        ax[1].set_ylabel(r"$Y_{GSM}$ / $R_E$", fontsize=15, labelpad=-2)
        ax[0].set_xlabel(r"$X_{GSM}$ / $R_E$", fontsize=15)
        ax[1].set_xlabel(r"$X_{GSM}$ / $R_E$", fontsize=15)
    else:
        ax[0].set_ylabel(r"$Z$ / $R_E$", fontsize=15, labelpad=-2)
        ax[1].set_ylabel(r"$Y$ / $R_E$", fontsize=15, labelpad=-2)
        ax[0].set_xlabel(r"$X$ / $R_E$", fontsize=15)
        ax[1].set_xlabel(r"$X$ / $R_E$", fontsize=15)

    plt.tight_layout()

    cbar_ax = fig.add_axes([1, 0.108, 0.016, 0.864])

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


def gen_seeds(plane, r_IB, xyz_sgn, mu, sim):
    """Generate seed points for streamlines.

    Args:
    ----
        plane (str): The plane in which to generate the seed points.
        Can be "xz" or "yz".
        r_IB (float): The radius of the sphere on which the seed points are generated.
        xyz_sgn (int): The sign of the x or y coordinate of the seed points.
        mu (float): The dipole tilt angle.
        sim (Simulation): The simulation object.

    Returns:
    -------
        tuple: A tuple containing the seed points as a numpy array of shape (n, 3) and
        the corresponding seed labels as a numpy array of shape (n,).

    """
    if plane == "xz":
        theta = 0
        dth = 4 * np.pi / 180
        b0 = np.sqrt(1 + 3 * np.cos(theta) ** 2)

        theta = theta + dth
        thetas = [0, theta]
        while theta < 80 * np.pi / 180:
            b = np.sqrt(1 + 3 * np.cos(theta) ** 2)
            theta = theta + (b0 / b) ** 3 * dth
            thetas.append(theta)
        thetas = np.array(thetas)
        thetas = np.append(-thetas[::-1][:-1], thetas)
        seed_labels = 0 * thetas
        thetas = np.append(thetas, np.pi - thetas)
        seed_labels = np.append(seed_labels, seed_labels + 1)
        x = r_IB * np.sin(thetas)
        z = r_IB * np.cos(thetas)
        x, _, z = rot_y(x, 0 * x, z, -mu)
        x_SW = np.arange(max(-22, sim.xb[0]), min(40, sim.xb[-1]), 4)
        x = np.append(x, x_SW)
        z = np.append(z, min(30, sim.yb[-1]) + 0 * x_SW)
        seed_labels = np.append(seed_labels, 0 * x_SW + 2)

        return np.vstack([x, 0 * z, z]).T, seed_labels
    else:
        xstrs, ystrs = np.array(
            [np.linspace(-22, 30.1, 10), np.linspace(-30, 30.1, 10)]
        )
        xstr, zstr = np.meshgrid(xstrs, ystrs)
        xstr, zstr = xstr.T, zstr.T
        seed_labels = 0 * xstr.ravel() + 3

        return (
            np.vstack(
                [xstr.ravel() * xyz_sgn, zstr.ravel() * xyz_sgn, 0 * zstr.ravel()]
            ).T,
            seed_labels,
        )


def IB_mask(
    radius,
    center=(0, 0),
    plane="xz",
    zangle=90,
    ax=None,
    lw=0.5,
    color="k",
):
    """Add a mask to the inner boundary.

    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    from matplotlib.patches import Circle, Wedge

    if ax is None:
        ax = plt.gca()

    theta1, theta2 = zangle, zangle + 180
    if color == "k":
        circcolor = "w"
    else:
        circcolor = "k"

    if plane in ("xy", "xz"):
        zorder = 3
    elif plane == "yz":
        zorder = 1
    c1 = Circle(center, radius, fc=color)
    c2 = Circle(center, 1, edgecolor=circcolor, fc=color, lw=lw, zorder=2)
    w1 = Wedge(center, 1, theta1, theta2, fc="w", zorder=zorder)
    w2 = Wedge(center, 1, theta2, theta1, fc="k", zorder=zorder)
    for shape in [c1, c2, w1, w2]:
        ax.add_artist(shape)

    return [w1, w2]
