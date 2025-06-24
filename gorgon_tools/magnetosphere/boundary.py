"""Module for the fluopause class."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import seaborn as sns
from scipy import interpolate as interp
from scipy.interpolate import RegularGridInterpolator as interp3d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class fluopause:
    """Class for the fluopause surface."""

    def __init__(self, nyz=200, nth=48, nu=40):
        """Initialise the fluopause class."""
        self.nyz = nyz
        self.nth, self.nu = nth, nu

    def calc(self, sim, debug=False):
        """Todo: Docstring for calc."""
        # Get required arrays
        for s in ["B{}_c", "v{}", "j{}"]:
            if s.format("vec") in sim.arr:
                sim.arr[s[0] + "mag"] = np.sqrt(
                    np.sum(sim.arr[s.format("vec")] ** 2, axis=-1)
                )

        # Seed points for streamlines (slightly randomised for better statistics)
        nz, ny = self.nyz, self.nyz
        z0 = np.linspace(-1, 1, nz) * 10
        y0 = np.linspace(-1, 1, ny) * 10
        dy, dz = z0[1] - z0[0], y0[1] - y0[0]
        z0, y0 = np.meshgrid(z0, y0)
        x0 = -20 * np.ones_like(z0)
        z0 += dz * (np.random.rand(nz, ny) - 0.5)
        y0 += dy * (np.random.rand(nz, ny) - 0.5)
        xs = np.vstack([x0.ravel(), y0.ravel(), z0.ravel()]).T

        # Generate streamlines
        from .streamline import streamline_array

        streams = streamline_array(10000, 0.1 * np.mean(sim.dx), 1)
        streams.calc(
            xs,
            sim.arr["vvec"],
            d=np.ones([3]) * np.mean(sim.dx),
            xc=sim.center - 0.5 * np.mean(sim.dx),
            v_name="vvec",
        )

        # Check this worked
        if debug:

            def latex(x):
                return "$\mathsf{" + x + "}$"

            sns.set_context("notebook")
            sns.set_palette("colorblind")
            sns.set_style("ticks")
            plt.rcParams["image.cmap"] = "viridis"
            fig, ax = plt.subplots()
            iy = len(sim.yb) // 2
            ax.pcolormesh(sim.xb, sim.zb, sim.arr["rho"][:, iy, :].T)
            step = max(1, len(streams.xs) // 177)
            for xs in streams.xs[::step]:
                ax.plot(xs[:, 0], xs[:, 2], color="w", alpha=0.25)

        # Get density along streamlines and mask out low-density regions
        streams.interp(sim.xc, sim.yc, sim.zc, sim.arr["rho"], "rho")
        rho_min = np.array([np.nanmin(rho) for rho in streams.var["rho"]])
        rho_min = rho_min[np.isfinite(rho_min)]
        streams.xs = streams.xs[rho_min > 1e-21]

        # Convert streamline coordinates into parabolic coordinates
        xs = np.vstack(streams.xs).astype(float)
        us = np.array(self.cart_to_par(xs[:, 1], xs[:, 2], xs[:, 0])).T
        u_c = np.linspace(0, 1, self.nu) * np.nanmax(us[:, 0])
        th_c = np.linspace(-1, 1, self.nth + 1)[1:] * np.pi

        # Generate coordinate bins in which to find density minima
        du, dth = u_c[1] - u_c[0], th_c[1] - th_c[0]
        th_e = th_c - 0.5 * dth
        th_e = np.hstack([th_e, th_e[-1] + dth])
        u_e = u_c - 0.5 * du
        u_e = np.hstack([u_e, u_e[-1] + du])

        # Extract 'void' locations in streamline density
        from scipy.stats import binned_statistic_2d

        V_c = binned_statistic_2d(
            us[:, 0], us[:, 2], us[:, 1], statistic=np.nanmin, bins=(u_e, th_e)
        )[0].T
        V_c[:, 0] = self.reduce(V_c[:, 0])

        # Filter out some noise
        from scipy.ndimage.filters import gaussian_filter

        V_c = gaussian_filter(V_c, 0.1)

        # Check this worked (should look like a smooth distriution)
        if debug:
            fig, ax = plt.subplots()

            p = ax.pcolormesh(np.degrees(th_e), u_e, V_c.T)
            cbar = plt.colorbar(p, ax=ax)
            cbar.ax.set_title("v")
            ax.set(
                xlabel=latex("\\phi\;(^o)"),
                ylabel=latex("u"),
                xlim=[-180, 180],
            )
            plt.show()

        # From 'void' regions, get magnetopause surface in parabolic and cartesian
        # coordinates
        U_c, TH_c = np.meshgrid(u_c, th_c)
        Y_c, Z_c, X_c = self.par_to_cart(U_c, V_c, TH_c)

        # Check this worked (should see a smooth FP surface in noon-midnight plane)
        if debug:
            i = [np.abs(th_c) == 0.5 * np.pi]
            _, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
            iy = len(sim.yc) // 2
            p = ax[0].pcolormesh(
                sim.xc, sim.zc, np.log10(sim.arr["jmag"][:, iy, :].T), shading="auto"
            )
            p.set_rasterized(True)
            cbar = plt.colorbar(p, ax=ax[0])
            cbar.ax.set_title(latex("n\;(/cm^{-3})"))
            p = ax[1].pcolormesh(
                sim.xc, sim.zc, np.log10(sim.arr["jmag"][:, iy, :].T), shading="auto"
            )
            p.set_rasterized(True)
            cbar = plt.colorbar(p, ax=ax[1])
            cbar.ax.set_title(latex("|j|\;(nA/m^2)"))
            for xs in streams.xs:
                us = np.array(
                    self.cart_to_par(
                        xs[:, 1].astype(float),
                        xs[:, 2].astype(float),
                        xs[:, 0].astype(float),
                    )
                ).T
                if (np.abs(np.abs(us[:, 2]) - 0.5 * np.pi) < 0.5 * dth).any():
                    ax[0].plot(xs[:, 0], xs[:, 2], color="w", alpha=0.25)
            for axi, c in zip(ax, [":k", ":w"]):
                for ii in i:
                    axi.plot(X_c[ii, :].T, Z_c[ii, :].T, c)

                axi.set_xlim([sim.xc.min(), sim.xc.max() + 0.5 * sim.dx[0]])
                axi.set_ylim([sim.yc.min(), sim.yc.max() + 0.5 * sim.dx[1]])
                axi.set_xticklabels([f"{-xi:2.0f}" for xi in axi.get_xticks()])

                axi.set_xlabel(latex("x_{GSE}\;(R_E)"))
            ax[0].set_ylabel(latex("z_{GSE}\;(R_E)"))

            plt.show()

        # Store key fluopause parameters
        self.th, self.u = th_c, u_c
        self.X, self.Y, self.Z = X_c, Y_c, Z_c
        self.X_cyc, self.Y_cyc, self.Z_cyc = self.cyclical(self.X, self.Y, self.Z)

    def calc_normals(self):
        """Must be cyclical about the first axis (i.e. first value = last value)."""
        X, Y, Z = self.X, self.Y, self.Z

        # Create empty normals array
        FP_norm = np.zeros([3, len(X[:, 0]), len(X[0, :])])

        # Need to fill any NaNs for statistics to work
        X[np.isnan(X)], Y[np.isnan(Y)], Z[np.isnan(Y)] = 100, 100, 100

        # Add extra points to arrays for method to work
        X, Y, Z = (
            np.append(X, [X[1, :]], axis=0),
            np.append(Y, [Y[1, :]], axis=0),
            np.append(Z, [Z[1, :]], axis=0),
        )
        X, Y, Z = (
            np.append([X[-3, :]], X, axis=0),
            np.append([Y[-3, :]], Y, axis=0),
            np.append([Z[-3, :]], Z, axis=0),
        )

        # Loop through and calculate normals using eigenvector method (see Komar et al.
        # (2015), JGR)
        for i in range(1, len(X[:, 0]) - 1):
            for j in range(1, len(X[0, :]) - 1):
                Xk = [X[i - 1, j], X[i, j], X[i + 1, j], X[i, j - 1], X[i, j + 1]]
                Yk = [Y[i - 1, j], Y[i, j], Y[i + 1, j], Y[i, j - 1], Y[i, j + 1]]
                Zk = [Z[i - 1, j], Z[i, j], Z[i + 1, j], Z[i, j - 1], Z[i, j + 1]]
                cov = np.cov(np.vstack([Xk, Yk, Zk]))
                evals, evecs = np.linalg.eig(cov)
                FP_norm[:, i - 1, j - 1] = (
                    -np.sign(evecs[:, np.argmin(evals)][0]) * evecs[:, np.argmin(evals)]
                )

            # Do the boundaries
            Xk = np.append(X[i, 0], X[:, 1])
            Yk = np.append(Y[i, 0], Y[:, 1])
            Zk = np.append(Z[i, 0], Z[:, 1])
            cov = np.cov(np.vstack([Xk, Yk, Zk]))
            evals, evecs = np.linalg.eig(cov)
            FP_norm[:, i - 1, 0] = evecs[:, np.argmin(evals)]

            Xk = np.append(X[i, -1], X[i, -2])
            Yk = np.append(Y[i, -1], Y[i, -2])
            Zk = np.append(Z[i, -1], Z[i, -2])
            cov = np.cov(np.vstack([Xk, Yk, Zk]))
            evals, evecs = np.linalg.eig(cov)
            FP_norm[:, i - 1, -1] = evecs[:, np.argmin(evals)]

        self.normals = FP_norm
        nx, ny, nz = self.cyclical(FP_norm[0, :, :], FP_norm[1, :, :], FP_norm[2, :, :])
        self.normals_cyc = np.array([nx, ny, nz])

    def cyclical(self, X, Y, Z):
        """Make the input arrays cyclical.

        Makes the input arrays cyclical by appending the first element to the end of
        each array.

        Args:
        ----
            X (numpy.ndarray): Array of X coordinates.
            Y (numpy.ndarray): Array of Y coordinates.
            Z (numpy.ndarray): Array of Z coordinates.

        Returns:
        -------
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple of the cyclical
            X, Y, and Z arrays.

        """
        X, Y, Z = (
            np.concatenate((X, np.array([X[0, :]])), axis=0),
            np.concatenate((Y, np.array([Y[0, :]])), axis=0),
            np.concatenate((Z, np.array([Z[0, :]])), axis=0),
        )
        return X, Y, Z

    def cart_to_par(self, x, y, z):
        """Convert Cartesian coordinates to parabolic coordinates.

        Args:
        ----
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate

        Returns:
        -------
        tuple: A tuple containing the u, v, and theta values in parabolic coordinates.

        """
        r = np.sqrt(x**2 + y**2 + z**2)

        th = np.arctan2(y, x)
        u = np.sqrt(r + z)
        v = np.sqrt(r - z)

        return u, v, th

    def par_to_cart(self, u, v, th):
        """Convert parabolic coordinates to Cartesian coordinates.

        Args:
        ----
            u (float): The u coordinate.
            v (float): The v coordinate.
            th (float): The theta coordinate.

        Returns:
        -------
            Tuple[float, float, float]: The x, y, and z coordinates in Cartesian space.

        """
        x = u * v * np.cos(th)
        y = u * v * np.sin(th)
        z = 0.5 * (u**2 - v**2)

        return x, y, z

    def write_vtp(self, fname):
        """Write the boundary data to a VTK file in VTP format.

        Args:
        ----
        fname (str): The name of the output file.

        Returns:
        -------
        None

        """
        import numpy as np
        import vtk

        x = np.vstack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()]).T

        def index_order(i, j):
            return [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]

        index = np.arange(len(x)).reshape(self.nth, self.nu)
        index = np.vstack([index[-1, :], index])
        face = np.array(
            [
                [index[s[0], s[1]] for s in index_order(i, j)]
                for i in range(len(self.th))
                for j in range(len(self.u) - 1)
            ]
        )

        def mkVtkIdList(it):
            """Make a vtkIdList from a Python iterable."""
            vil = vtk.vtkIdList()
            for i in it:
                vil.InsertNextId(int(i))
            return vil

        # We'll create the building blocks of polydata including data attributes.
        cube = vtk.vtkPolyData()
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        # Load the point, cell, and data attributes.
        for i, xi in enumerate(x):
            points.InsertPoint(i, xi)
        for i, pts_i in enumerate(face):
            polys.InsertNextCell(mkVtkIdList(pts_i))

        # We now assign the pieces to the vtkPolyData.
        cube.SetPoints(points)
        # del points
        cube.SetPolys(polys)
        # del polys

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fname)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()

    def reduce(self, v):
        """Return the minimum value in the input array, ignoring any NaN values.

        Parameters
        ----------
        v : numpy.ndarray
            The input array.

        Returns
        -------
        float
            The minimum value in the input array, ignoring any NaN values.

        """
        try:
            return np.nanmin(v)
        except ValueError:
            return np.NaN


def MP_mask(sim, FP, x_max=None, debug=False):
    """Mask the magnetosphere boundary in the simulation grid.

    Args:
    ----
    sim (Simulation): The simulation object.
    FP (fluopause): The fluopause object.
    x_max (float): The maximum X value to mask out.
    debug (bool): Whether to plot debug figures.

    Returns:
    -------
        None

    """
    X_MP_in, Y_MP_in, Z_MP_in = (
        FP.X,
        FP.Y,
        FP.Z,
    )  # dim1 is theta (clockwise in Y-Z vieing from Sun), dim2 is u
    # (~ increasing downtail)

    # Truncated magnetopause surface coordinates (xmax is limit of masking in X)
    if x_max is None:
        x_max = sim.xb[-1]
    ix = np.argmin(abs(X_MP_in[0, :] - x_max - 5))
    X_MP, Y_MP, Z_MP = X_MP_in[:, :ix], Y_MP_in[:, :ix], Z_MP_in[:, :ix]

    # We will iterate along the X axis and mask in Y and Z
    xs = (
        np.arange(
            np.floor(X_MP[0, 0] / np.min(sim.dx) + 1) * np.min(sim.dx),
            x_max,
            np.min(sim.dx),
        )
        + np.min(sim.dx) / 2
    )
    Ys, Zs = np.zeros([len(X_MP[:, 0]), len(xs)]), np.zeros([len(X_MP[:, 0]), len(xs)])

    # Since Y_MP and Z_MP are not defined for all X, we will need to interpolate to get
    # these - so create interpolation functions
    for i in range(len(Ys[:, 0])):
        int_Y = interp.interp1d(
            X_MP[i, :], Y_MP[i, :], bounds_error=False, fill_value="extrapolate"
        )
        int_Z = interp.interp1d(
            X_MP[i, :], Z_MP[i, :], bounds_error=False, fill_value="extrapolate"
        )
        Ys[i, :] = int_Y(xs)
        Zs[i, :] = int_Z(xs)

    # Indices
    jmin, jmax = (
        int(np.floor((np.min(Ys) + sim.center[1]) / np.min(sim.dy))),
        int(np.floor((np.max(Ys) + sim.center[1]) / np.min(sim.dy))),
    )
    kmin, kmax = (
        int(np.floor((np.min(Zs) + sim.center[2]) / np.min(sim.dz))),
        int(np.floor((np.max(Zs) + sim.center[2]) / np.min(sim.dz))),
    )

    # Iterate through grid and create a masking arrayg based on some Shapely magic...
    xc, yc, zc = sim.xc, sim.yc, sim.zc
    x, y, z = np.meshgrid(xc, yc, zc, indexing="ij")
    sim.arr["mask"] = 0 * sim.arr["rho"] + 1
    for i in range(len(xs)):
        ix = np.floor((xs[i] + sim.center[0]) / np.min(sim.dz)).astype(int)
        polygon = Polygon(zip(Ys[:, i], Zs[:, i]))
        for j in range(jmin, jmax + 1):
            for k in range(kmin, kmax + 1):
                point = Point(y[ix, j, k], z[ix, j, k])
                sim.arr["mask"][ix, j, k] = 1 - polygon.contains(point)

    # For those grid cells not used, set mask to zero
    sim.arr["mask"][int(np.floor((x_max + sim.center[0]) / np.min(sim.dz))) :, :, :] = 0


def cusp_mask(sim, r_IB, r_cusps, debug):
    """Create a mask for the cusps of the magnetosphere using simulation data.

    Args:
    ----
        sim (Simulation): The simulation data.
        r_IB (float): The inner boundary radius.
        r_cusps (float): The radius of the cusps.
        debug (bool): Whether to show debug plots or not.

    Returns:
    -------
        None

    """
    # Create an array of spherical polar coordinates on the dayside magnetosphere
    ths = np.arange(-90, 91, 1) * np.pi / 180  # Latitude (up-down)
    phs = (
        np.arange(120, 241, 1) * np.pi / 180
    )  # Longitude (left-right) - I picked this range because I found the cusps are
    # within this, but could be bigger
    th, ph = np.meshgrid(ths, phs)
    dr = np.min(sim.dx) / 2
    rs = np.arange(
        r_IB + dr, r_cusps, dr
    )  # We will go out from the inner boundary to r_cusps (should be roughly at the
    # magnetopause)

    for i in range(len(rs)):
        rsph = rs[i]

        # Convert to cartesian coordinates to get values from simulation data
        xsph = rsph * np.cos(th) * np.cos(ph)
        ysph = rsph * np.cos(th) * np.sin(ph)
        zsph = rsph * np.sin(th)

        # Trilinear interpolation to get sub-grid plasma pressure
        f = interp3d((sim.xc, sim.yc, sim.zc), sim.arr["P"])
        P = f((xsph, ysph, zsph))

        # Plot contour maps
        col_levs = np.arange(0, 1.3, 0.01)
        con_levs = [
            0.6 * np.max(P)
        ]  # Only draw the contour line where it drops-off to 50% of maximum pressure
        col = plt.contourf(
            ph * 180 / np.pi, th * 180 / np.pi, P / 1e-9, col_levs, cmap="jet"
        )
        con = plt.contour(ph * 180 / np.pi, th * 180 / np.pi, P, con_levs)
        cbar = plt.colorbar(col)
        cbar.set_label(r"$P$ / nPa", fontsize=12)

        # Extract the coordinates of the 50% contour
        pts = con.allsegs[0]
        lens = [len(arr) for arr in pts]
        cusp1, cusp2 = (
            pts[np.argsort(lens)[-2]],
            pts[np.argsort(lens)[-1]],
        )  # Might get some spurious contours so only pick the biggest one!
        if np.average(cusp1[:, 1]) > np.average(cusp2[:, 1]):
            N_cusp, S_cusp = cusp1 * np.pi / 180, cusp2 * np.pi / 180
        else:
            N_cusp, S_cusp = cusp2 * np.pi / 180, cusp1 * np.pi / 180

        N_lons, N_lats = N_cusp[:, 0], N_cusp[:, 1]
        S_lons, S_lats = S_cusp[:, 0], S_cusp[:, 1]

        N_poly = Polygon(zip(N_lons, N_lats))
        S_poly = Polygon(zip(S_lons, S_lats))

        mask = 0 * th.astype(bool)
        for i in range(len(phs)):
            for j in range(len(ths)):
                point = Point(ph[i, j], th[i, j])
                mask[i, j] = N_poly.contains(point) or S_poly.contains(point)

        ix = (
            np.floor(
                (rsph * np.cos(th) * np.cos(ph) * mask + sim.center[0]) / np.min(sim.dx)
            )
        ).ravel()
        iy = (
            np.floor(
                (rsph * np.cos(th) * np.sin(ph) * mask + sim.center[1]) / np.min(sim.dy)
            )
        ).ravel()
        iz = (np.floor((rsph * np.sin(th) * mask + sim.xc[2]) / np.min(sim.dz))).ravel()
        cusp_inds = np.unique(np.array([ix, iy, iz]), axis=1).astype(int)
        for i in range(len(cusp_inds[0, :])):
            ix, iy, iz = cusp_inds[0, i], cusp_inds[1, i], cusp_inds[2, i]
            sim.arr["mask"][ix, iy, iz] = 1

        if debug:
            plt.xlim(120, 240)
            plt.xlabel(r"Longitude / $^\circ$", fontsize=12)
            plt.ylabel(r"Latitude / $^\circ$", fontsize=12)
            plt.title(r"$r$ = " + "%.2f" % rsph + " " + "$R_E$")
            plt.ylim(-90, 90)
            plt.show()
        else:
            plt.close()


def find_BS(x, rho, drho_min=0.05, debug=False):
    """Find the bow shock location in the magnetosphere.

    Finds the bow shock (BS) location in the magnetosphere given the radial distance and
    density profile.

    Args:
    ----
        x (array-like): Radial distance array.
        rho (array-like): Density profile array.
        drho_min (float, optional): Minimum density gradient required to identify
        the BS. Defaults to 0.05.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
    -------
        float: Bow shock location.

    """
    x_BS, i_BS = find_BS_cell(x, rho, drho_min, debug=debug)
    if i_BS == -99:
        return x[0]
    else:
        return BS_sg_parabolic(x, rho, i_BS, debug=debug)


def gradient(y, x):
    """Compute the gradient of a function y(x).

    Compute the gradient of a function y(x) using a second-order accurate
    finite difference scheme.

    Args:
    ----
        y (ndarray): The function values at the grid points.
        x (ndarray): The grid points.

    Returns:
    -------
        tuple: A tuple containing the gradient values and the cell-centered grid points.

    """
    dy = y[2:] - y[1:-1]
    dy2 = y[2:] - y[:-2]
    dy = (27 * dy - dy2) / 24
    xc = 0.5 * (x[2:] + x[1:-1])

    return dy, xc


def find_BS_cell(x, rho, drho_min, debug=False):
    """Find the furthest shock in a given x line based on the mass density gradient.

    Args:
    ----
        x (numpy.ndarray): The x line.
        rho (numpy.ndarray): The mass density values along the x line.
        drho_min (float): The minimum value of the mass density gradient to consider a
        peak significant.
        debug (bool, optional): Whether to plot debug information. Defaults to False.

    Returns:
    -------
        tuple: The x coordinate and index of the furthest shock.

    """
    from scipy.signal import argrelmax

    if debug:
        import matplotlib.pyplot as plt

    """ Takes values along the x line """
    # Calculate mass density gradient
    drho, xc = gradient(rho, x)

    """ Normalise drho"""
    C = drho.max()
    drho = drho / C

    """ Plot """
    if debug:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(x, rho, ".-")
        ax[1].plot(xc, drho, ".-")

    """ Find negative peaks """
    pks = argrelmax(drho)[0]
    if pks.shape[0] == 0:
        print("BS: No peaks found")
        return np.nan, -99

    if debug:
        for i in pks:
            for axi in ax:
                axi.axvline(x=xc[i], linestyle="dashed", color="k")

    """ Sort by descending distance """
    s = np.argsort(xc[pks])
    pks = pks[s]

    """ Remove insignificant peaks"""
    el = drho[pks] > drho_min
    pks = pks[el]

    if pks.shape[0] == 0:
        print("BS: No significant peaks found")
        return np.nan, -99

    if debug:
        for i in pks:
            for axi in ax:
                axi.axvline(x=xc[i], linestyle="dashed", color="b")
        ax[1].axhline(y=drho_min, linestyle="dashed", color="k")

    """ Pick furthest shock """
    i_BS = pks[0]

    if debug:
        for axi in ax:
            axi.axvline(x=xc[i_BS], color="r")

        print(pks[0], xc[pks[0]])
    return xc[i_BS], i_BS


def BS_sg_parabolic(x, rho, i_BS, debug=False):
    """Calculate parabolic estimate of the magnetic field strength.

    Calculate the parabolic estimate of the magnetic field strength at the boundary of
    the magnetosphere.

    Args:
    ----
        x (numpy.ndarray): The x-coordinates of the data points.
        rho (numpy.ndarray): The magnetic field strength at each x-coordinate.
        i_BS (int): The index of the boundary point.
        debug (bool, optional): If True, plots the data and the parabolic estimate.
        Defaults to False.

    Returns:
    -------
        float: The x-coordinate of the parabolic estimate of the magnetic field strength
        at the boundary.

    """
    if debug:
        import matplotlib.pyplot as plt
    """ Calculate gradient again"""
    dy, xc = gradient(rho, x)

    if debug:
        """Plot"""
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(x, rho, ".-")
        ax[0].set_ylabel("f")
        ax[1].plot(xc, dy, ".-")
        ax[1].set_ylabel("$\partial_x$ f")

    """ Select points around x_BS_grid"""
    di = 1

    dyi = dy[i_BS - di : i_BS + di + 1]
    xci = xc[i_BS - di : i_BS + di + 1]

    if debug:
        ax[1].plot(xci, dyi, "r.")

    """ Parabolic estimate """
    x2 = xci * xci

    a = (
        dyi[1] * (xci[2] - xci[0])
        - dyi[0] * (xci[2] - xci[1])
        - dyi[2] * (xci[1] - xci[0])
    )
    a /= (
        x2[0] * (xci[1] - xci[2])
        - x2[2] * (xci[1] - xci[0])
        - x2[1] * (xci[0] - xci[2])
    )

    b = dyi[1] - dyi[0] + a * (x2[0] - x2[1])
    b /= xci[1] - xci[0]

    x_BS = -0.5 * b / a

    if debug:
        """Plot"""
        for axi in ax:
            axi.axvline(x=x_BS, linestyle="dashed", color="r")
            axi.axvline(x=xci[1], linestyle="dashed", color="g")
        xi = np.linspace(xci[0], xci[-1])
        c = dyi[0] - a * xci[0] ** 2 - b * xci[0]
        dyi = a * xi**2 + b * xi + c
        ax[1].plot(xi, dyi, "r")
        ax[0].set_xlim(xc[i_BS - 5], xc[i_BS + 5])
        plt.show()

    return x_BS


def find_MP(x, j, rho, debug=False):
    """Find the magnetopause (MP) location and current density.

    Find the magnetopause (MP) location and current density from input arrays of
    position, current density, and mass density.

    Args:
    ----
        x (numpy.ndarray): Array of positions.
        j (numpy.ndarray): Array of current densities.
        rho (numpy.ndarray): Array of mass densities.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
    -------
        tuple: A tuple containing the MP location and current density as floats.

    """
    dx = x[1] - x[0]
    if debug:
        print(f"Input array shapes: x {x.shape}, j {j.shape}, rho {rho.shape}")
    drho = np.gradient(rho, dx)

    # Find Peaks
    pks = sig.argrelmax(j)[0]
    if debug:
        print(f"No. of peaks in j: {pks.shape}")

    # Ensure mass decreases over MP
    el_rho = drho[pks] >= 0
    pks = pks[el_rho]
    if debug:
        print(f"No. of peaks after rho test {pks.shape}")

    # Choose largest current density
    # s = np.argsort(j[pks])
    s = np.argsort(drho[pks])
    pks = pks[s]

    i_MP = pks[-1]

    # Estimate sub-grid position using parabola
    x_MP = parabolic_estimate(x[i_MP - 1 : i_MP + 2], j[i_MP - 1 : i_MP + 2])
    j_MP = j[i_MP]

    return x_MP, j_MP


def find_MP_old(x, j, rho, B, P_T, x_BS=None, show_errors=True):
    """Find the magnetopause (MP) position using the provided input parameters.

    Args:
    ----
        x (numpy.ndarray): The x-coordinate array.
        j (numpy.ndarray): The current density array.
        rho (numpy.ndarray): The mass density array.
        B (numpy.ndarray): The magnetic field array.
        P_T (numpy.ndarray): The total pressure array.
        x_BS (float, optional): The position of the bow shock. Defaults to None.
        show_errors (bool, optional): Whether to show error messages. Defaults to True.

    Returns:
    -------
        float: The position of the magnetopause.

    """
    # j = line['j']
    # rho = line['rho']
    # B = line['B']
    # P_T = line['P_T']

    # Calculate
    i = sig.argrelmax(j)[0]

    if len(i) == 0:
        return (np.nan, np.nan)

    """ Sort by descending distance """
    s = np.argsort(x[i])
    i = i[s]

    # ''' Sort by largest j peak '''
    # s = np.argsort(j[i])
    # s = s[::-1]
    # i = i[s]

    """ Remove values near edges """
    el = np.logical_and(2 < i, i < len(j) - 2)
    i = i[el]

    """ Test other properties:
         - B increases across the magnetopause
         - rho, mom decreases across the magnetopause """

    i0 = i - 2
    i1 = i + 2

    el_B = B[i0] < B[i1]
    el_rho = rho[i0] > rho[i1]
    el_Pbal = P_T[i0] > P_T[i1]
    el_rho2 = rho[i] > 1e-23

    if x_BS is None:  # Remove peaks outside bow shock
        el_BS = True
    else:
        el_BS = np.abs(x[i]) < np.abs(x_BS - 1.0)

    el = np.logical_and(el_B, el_rho)
    el = np.logical_and(el, el_rho2)
    el = np.logical_and(el, el_Pbal)
    el = np.logical_and(el, el_BS)

    """ Store value """
    i = i[el]

    if len(i) == 0:
        if show_errors:
            print("Error: No MP found")
        return np.nan  # (np.nan,np.nan)

    i = i[
        0
    ]  # If multiple values, take the largest gradient element: i[0] (since its sorted)

    """ Sub-grid BS position using parabola """
    x_MP_SG = parabolic_estimate(x[i - 1 : i + 2], j[i - 1 : i + 2])

    return x_MP_SG


def parabolic_estimate(x, y):
    """Estimate the location of the maximum value of a parabolic curve.

    Args:
    ----
        x (list): A list of three x-coordinates.
        y (list): A list of three y-coordinates.

    Returns:
    -------
        float: The x-coordinate of the maximum value of the parabolic curve.

    """
    x2 = x * x

    a = y[1] * (x[2] - x[0]) - y[0] * (x[2] - x[1]) - y[2] * (x[1] - x[0])
    a /= x2[0] * (x[1] - x[2]) - x2[2] * (x[1] - x[0]) - x2[1] * (x[0] - x[2])

    b = y[1] - y[0] + a * (x2[0] - x2[1])
    b /= x[1] - x[0]

    return -b / (2 * a)
