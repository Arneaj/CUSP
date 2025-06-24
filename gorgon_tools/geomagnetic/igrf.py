"""Functions pertaining to the International Geomagnetic Reference Field model."""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coordinates import sph_to_cart_vec


def IGRF_load_coeffs(time, order):
    """Load the IGRF coefficients for a given time and order.

    Args:
    ----
        time (datetime.datetime or list): The time(s) for which to load
        the coefficients.
        order (int): The order of the coefficients to load.

    Returns:
    -------
        tuple: A tuple containing two numpy arrays, g and h, which contain the spherical
        harmonic coefficients for the magnetic field.

    """
    from io import BytesIO
    from pkgutil import get_data

    data = get_data(__name__, "data/igrf13coeffs.csv")
    dat = np.genfromtxt(BytesIO(data), skip_header=3, delimiter=",", dtype=str).T
    ghmn = []
    for i in range(1, len(dat[0, :])):
        ghmn.append(dat[0, i] + dat[1, i] + dat[2, i])
    igrf_dat = pd.DataFrame(
        dat[3:, 1:].astype(float),
        columns=ghmn,
        index=[dt.datetime(y, 1, 1) for y in dat[3:, 0].astype(int)],
    )
    igrf_dat = igrf_dat.resample("1ME").mean().interpolate(method="linear")

    if isinstance(time, list):
        time = np.array(time)
    elif isinstance(time, dt.datetime):
        time = np.array([time])

    coeffs = igrf_dat.iloc[
        igrf_dat.index.get_indexer([pd.to_datetime(time[0])], method="nearest")
    ]
    order = 5
    g = np.zeros([order, order + 1])
    h = np.zeros([order, order + 1])
    for n in range(0, order):
        for m in range(order + 1):
            if m <= n + 1:
                g[n, m] = float(coeffs["g" + str(n + 1) + str(m)].iloc[0])
                if m > 0:
                    h[n, m] = float(coeffs["h" + str(n + 1) + str(m)].iloc[0])

    return g, h


def IGRF_calc(r, th, az, time, order):
    """Calculate the magnetic field vector using the IGRF model.

    Args:
    ----
        r (float): radial distance from the center of the Earth in kilometers
        th (float): colatitude (polar angle) in radians
        az (float): azimuth (longitude) in radians
        time (float): decimal year representing the date and time of the observation
        order (int): maximum degree and order of the IGRF model coefficients to use

    Returns:
    -------
        tuple: a tuple containing the magnetic field vector components in the radial,
        colatitudinal, and azimuthal directions, respectively, in nanotesla (nT)

    """
    from ._fortran import igrf_subs

    g, h = IGRF_load_coeffs(time, order)

    Br, Bth, Baz = igrf_subs.igrf_grid(r, th, az, g, h)

    return Br, Bth, Baz


def IGRF_map(r, th, az, time, r_map=1, order=6, d=0.045, disp=False):
    """Map the International Geomagnetic Reference Field (IGRF) to a spherical surface.

    Args:
    ----
        r (float): Radial distance from the center of the Earth.
        th (float): Polar angle in radians.
        az (float): Azimuthal angle in radians.
        time (float): Decimal year specifying the time of the IGRF model.
        r_map (float, optional): Radial distance of the mapping surface. Defaults to 1.
        order (int, optional): Order of the IGRF model. Defaults to 6.
        d (float, optional): Tracing step size for field line tracing.
        Defaults to 0.045.
        disp (bool, optional): Whether to display a 3D plot of the field lines.
        Defaults to False.

    Returns:
    -------
        tuple: A tuple containing the polar and azimuthal angles of the mapped
        field lines.

    """
    from ..magnetosphere.fieldlines import fieldlines

    d_trace = d / 2  # tracing step size
    n_steps = int(
        float(3 * r) / d_trace
    )  # Number of steps available to the streamline solver
    flines = fieldlines(n_steps, d_trace)

    # Sample direction of field at each point, used for mapping
    r_seeds, th_seeds, az_seeds = 0 * th.ravel() + r, th.ravel(), az.ravel()
    Br_seeds, _, _ = IGRF_calc(r_seeds, th_seeds, az_seeds, time, order)
    dirs = np.where(Br_seeds < 0, 1, -1)

    # Seed points for field line tracing
    x_seeds = r_seeds * np.sin(th_seeds) * np.cos(az_seeds)
    y_seeds = r_seeds * np.sin(th_seeds) * np.sin(az_seeds)
    z_seeds = r_seeds * np.cos(th_seeds)
    seeds = np.stack([x_seeds, y_seeds, z_seeds]).T

    # Trace field lines through a background IGRF field
    B = IGRF_grid(r, d, time, order)  # calculate background field
    flines.calc(seeds, B, np.array([d, d, d]), np.array([r, r, r]))  # trace field lines
    if disp:  # Optional plotting of field lines
        fig, ax = flines.plot3D()
        fig.set_size_inches(10, 10)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        plt.show()

    # Include only the Earthward section of the field lines, extract final point
    maps = 0 * seeds
    for i in range(len(flines.xs)):
        if dirs[i] == 1:
            flines.xs[i] = flines.xs_f[i].astype(float)
        else:
            flines.xs[i] = flines.xs_r[i].astype(float)
        maps[i] = flines.xs[i][
            np.argmin(abs(r_map - np.linalg.norm(flines.xs[i], axis=1)))
        ]

    # Transform back into geomagnetic coordinates
    th_map = np.arccos(maps[:, 2] / np.linalg.norm(maps, axis=1))
    az_map = np.arctan2(maps[:, 1], maps[:, 0])

    return th_map, az_map


def IGRF_grid(dim, d, time, order=6, disp=False):
    """Compute the International Geomagnetic Reference Field (IGRF) at a grid of points.

    Parameters
    ----------
    dim : float
        Half-dimension of the grid in units of Earth radii.
    d : float
        Grid spacing in units of Earth radii.
    time : float
        Decimal year for which to compute the IGRF.
    order : int, optional
        Maximum degree of the IGRF model to use. Default is 6.
    disp : bool, optional
        Whether to display plots of the magnetic field components. Default is False.

    Returns
    -------
    B : ndarray
        Array of shape (N, N, N, 3) containing the magnetic field vectors at each grid
        point, where N = 2*dim/d + 1.

    Notes
    -----
    This function uses the IGRF model to compute the Earth's magnetic field at a grid of
    points in a Cartesian coordinate system centered at the Earth's center. The magnetic
    field is expressed as a vector with components (Bx, By, Bz) in the Cartesian
    coordinate system.

    The IGRF model is a mathematical representation of the Earth's magnetic field that
    is based on measurements from ground-based observatories and satellites. It is
    updated every five years to account for changes in the Earth's magnetic field.

    References
    ----------
    [1] Thébault, E., Finlay, C. C., Beggan, C. D., Alken, P., Aubert, J., Barrois, O.,
        Bertrand, F., Bondar, T., Boness, A., Brocco, L., Canet, E., Chambodut, A.,
        Chulliat, A., Coïsson, P., Civet, F., Du, A., Fournier, A., Fratter, I.,
        Gillet, N., Hamilton, B., Hamoudi, M., Hulot, G., Jager, T., Korte, M.,
        Kuang, W., Lalanne, X., Langlais, B., Léger, J.-M., Lesur, V., Lowes, F. J.,
        Macmillan, S., Mandea, M., Manoj, C., Maus, S., Olsen, N., Petrov, V.,
        Ridley, V., Rother, M., Sabaka, T. J., Saturnino, D., Schachtschneider, R.,
        Sirol, O., Tangborn, A., Thomson, A., Tøffner-Clausen, L., Vigneron, P.,
        Wardinski, I., Zvereva, T. (2015). International Geomagnetic Reference Field:
        the 12th generation. Earth, Planets and Space, 67(1), 79.
        https://doi.org/10.1186/s40623-015-0228-9

    """
    xs, ys, zs = (
        np.arange(-dim, dim, d),
        np.arange(-dim, dim, d),
        np.arange(-dim, dim, d),
    )
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(z / r)
    az = np.arctan2(y, x)
    Br, Bth, Baz = IGRF_calc(r.ravel(), th.ravel(), az.ravel(), time, order)
    Bx, By, Bz = sph_to_cart_vec(Br, Bth, Baz, th.ravel(), az.ravel())
    Bx, By, Bz = (
        np.reshape(Bx, r.shape),
        np.reshape(By, r.shape),
        np.reshape(Bz, r.shape),
    )
    B = np.zeros([x.shape[0], x.shape[1], x.shape[2], 3])
    B[:, :, :, 0], B[:, :, :, 1], B[:, :, :, 2] = np.array([Bx, By, Bz])

    if disp:
        Bx[r < 1], By[r < 1], Bz[r < 1] = 0, 0, 0
        from matplotlib.patches import Circle

        _, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].contourf(
            x[:, :, len(zs) // 2], y[:, :, len(zs) // 2], Bx[:, :, len(zs) // 2], 50
        )
        ax[1].contourf(
            x[:, :, len(zs) // 2], y[:, :, len(zs) // 2], By[:, :, len(zs) // 2], 50
        )
        ax[2].contourf(
            x[:, :, len(zs) // 2], y[:, :, len(zs) // 2], Bz[:, :, len(zs) // 2], 50
        )
        ax[0].set_title("X-Y plane, Bx")
        ax[1].set_title("X-Y plane, By")
        ax[2].set_title("X-Y plane, Bz")
        for axi in ax:
            c1 = Circle((0, 0), 1.05, fc="k")
            axi.add_artist(c1)
            axi.set_xlabel(r"X / $R_E$")
            axi.set_ylabel(r"Y / $R_E$")
        _, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].contourf(
            x[:, len(ys) // 2, :], z[:, len(ys) // 2, :], Bx[:, len(ys) // 2, :], 50
        )
        ax[1].contourf(
            x[:, len(ys) // 2, :], z[:, len(ys) // 2, :], By[:, len(ys) // 2, :], 50
        )
        ax[2].contourf(
            x[:, len(ys) // 2, :], z[:, len(ys) // 2, :], Bz[:, len(ys) // 2, :], 50
        )
        ax[0].set_title("X-Z plane, Bx")
        ax[1].set_title("X-Z plane, By")
        ax[2].set_title("X-Z plane, Bz")
        for axi in ax:
            c1 = Circle((0, 0), 1.05, fc="k")
            axi.add_artist(c1)
            axi.set_xlabel(r"X / $R_E$")
            axi.set_ylabel(r"Z / $R_E$")
        _, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].contourf(
            y[len(xs) // 2, :, :], z[len(xs) // 2, :, :], Bx[len(xs) // 2, :, :], 50
        )
        ax[1].contourf(
            y[len(xs) // 2, :, :], z[len(xs) // 2, :, :], By[len(xs) // 2, :, :], 50
        )
        ax[2].contourf(
            y[len(xs) // 2, :, :], z[len(xs) // 2, :, :], Bz[len(xs) // 2, :, :], 50
        )
        ax[0].set_title("Y-Z plane, Bx")
        ax[1].set_title("Y-Z plane, By")
        ax[2].set_title("Y-Z plane, Bz")
        for axi in ax:
            c1 = Circle((0, 0), 1.05, fc="k")
            axi.add_artist(c1)
            axi.set_xlabel(r"Y / $R_E$")
            axi.set_ylabel(r"Z / $R_E$")
        plt.show()

    return B


def plot_IGRF(order, time, th_bound=20):
    """Plot the IGRF model for a given order and time.

    Args:
    ----
        order (int): The order of the IGRF model.
        time (float): The decimal year for which to calculate the IGRF model.
        th_bound (int, optional): The polar angle boundary in degrees. Defaults to 20.

    Returns:
    -------
        None

    """
    th = np.arange(th_bound * np.pi / 180, np.pi - th_bound * np.pi / 180, 0.01)
    az = np.arange(-np.pi - 0.01, np.pi, 0.01)
    th, az = np.meshgrid(th, az, indexing="ij")
    r = 0 * th + 1
    Br, Bth, Baz = IGRF_calc(r.ravel(), th.ravel(), az.ravel(), time, order)
    Br, Bth, Baz = (
        np.reshape(Br, r.shape),
        np.reshape(Bth, r.shape),
        np.reshape(Baz, r.shape),
    )
    B = np.linalg.norm(np.array([Br, Bth, Baz]), axis=0)
    plt.contourf(az * 180 / np.pi, th * 180 / np.pi, B, 50, cmap="RdBu_r")
    plt.contour(az * 180 / np.pi, th * 180 / np.pi, B, 15, linewidths=1, colors="k")
    plt.ylim(180 - th_bound, th_bound)
    plt.xlim(-180, 180)
    plt.title("Order = " + str(order))
    plt.show()
