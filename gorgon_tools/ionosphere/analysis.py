"""Module containing functions for analysing ionospheric data."""
import numpy as np


def calc_TFAC(iono, output=False):
    """Calculate the total TFAC on the polar cap.

    Calculate the total (1/2 * upward + downward) field-aligned current (TFAC) on the
    polar cap. Performs a surface integral on the FAC for each ionospheric grid cell.
    The resulting magnitudes are summed to obtain a value on each hemisphere.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        output (bool, optional): Choose whether to print out calculated values. Defaults
        to False.

    Returns:
    -------
        (float, float): Northern and southern TFAC values.

    """
    colat, _ = np.meshgrid(iono.th[:-1], iono.az)
    dth, daz = np.meshgrid(iono.dth, iono.daz_cyc)
    NFAC = 1 * iono.arr["FAC"][:-1, :].T
    NFAC[:, len(iono.th) // 2 :] = 0
    TFAC_N = np.sum(np.abs(NFAC) * iono.r_IS**2 * np.sin(colat) * daz * dth, (1, 0)) / 2

    SFAC = 1 * iono.arr["FAC"][:-1, :].T
    SFAC[:, : len(iono.th) // 2] = 0
    TFAC_S = np.sum(np.abs(SFAC) * iono.r_IS**2 * np.sin(colat) * daz * dth, (1, 0)) / 2

    if output:
        print("Northern Hemisphere: TFAC =", "%.1f" % (TFAC_N / 1e6), "\bMA")
        print("Southern Hemisphere: TFAC =", "%.1f" % (TFAC_S / 1e6), "\bMA")

    return TFAC_N, TFAC_S


def calc_CPCP(iono, output=False):
    """Calculate the cross-polar cap potential (CPCP) on each hemisphere.

    The CPCP is calculated as the difference between the minimum and maximum values of
    the electrostatic potential for a given hemisphere.

    Args:
    ----
        iono (ionosphere): Gorgon ionosphere object.
        output (bool, optional): Choose whether to print out calculated values.
        Defaults to False.

    Returns:
    -------
        (float, float): Northern and southern CPCP values.

    """
    Nphi = 1 * iono.arr["phi"].T
    Nphi[:, len(iono.th) // 2 :] = 0
    CPCP_N = np.max(np.max(Nphi)) - np.min(np.min(Nphi))

    Sphi = 1 * iono.arr["phi"].T
    Sphi[:, : len(iono.th) // 2] = 0
    CPCP_S = np.max(np.max(Sphi)) - np.min(np.min(Sphi))

    if output:
        print("Northern Hemisphere: CPCP =", "%.1f" % (CPCP_N / 1e3), "\bkV")
        print("Southern Hemisphere: CPCP =", "%.1f" % (CPCP_S / 1e3), "\bkV")

    return CPCP_N, CPCP_S


def calc_Evec(iono):
    """Calculate the electric field vector from the gradient of the potential phi.

    Args:
    ----
    iono (Ionosphere) : An object containing ionospheric data

    Returns:
    -------
    None

    """
    from ..geomagnetic.coordinates import sph_to_cart_vec

    if "Evec" not in iono.arr.keys():
        az, th = np.meshgrid(
            iono.az, iono.th, indexing="ij"
        )  # assuming a regular 2D grid for thin shell

        E_az, E_th = np.gradient(
            -iono.arr["phi"].T, iono.az, iono.th, axis=(0, 1)
        )  # get e-field from grad(phi)
        E_th = E_th / iono.r_IS  # scale grad correctly
        E_az[:, 1:-1] = E_az[:, 1:-1] / (iono.r_IS * np.sin(th[:, 1:-1]))
        E_x, E_y, E_z = sph_to_cart_vec(0 * E_th, E_th, E_az, th, az)
        E_x[:, 0], E_y[:, 0], E_z[:, 0] = (
            np.mean(E_x[:, 1]),
            np.mean(E_y[:, 1]),
            np.mean(E_z[:, 1]),
        )
        E_x[:, -1], E_y[:, -1], E_z[:, -1] = (
            np.mean(E_x[:, -2]),
            np.mean(E_y[:, -2]),
            np.mean(E_z[:, -2]),
        )
        iono.arr["Evec"] = np.swapaxes(np.array([E_x, E_y, E_z]), 0, 2)
        iono.arr["Evec"][0, :, 0] = np.mean(iono.arr["Evec"][1, :, 0])
        iono.arr["Evec"][-1, :, :] = np.mean(iono.arr["Evec"][-2, :, :], axis=0)
    else:
        print(
            "Evec already imported for timestep. Delete existing array "
            "if you wish to recalculate."
        )


def calc_JH(iono):
    """Todo: Fill this function description.

    Args:
    ----
        iono (Ionosphere): An instance of the Ionosphere class.

    Returns:
    -------
        None

    """
    if "Evec" not in iono.arr.keys():
        calc_Evec(iono)

    _, th = np.meshgrid(iono.az, iono.th)  # assuming a regular 2D grid for thin shell
    iono.arr["JH"] = np.linalg.norm(iono.arr["Evec"], axis=2) ** 2 * iono.arr["sig_P"]
    dth, daz = np.meshgrid(iono.dth, iono.daz_cyc, indexing="ij")
    iono.JH_N = np.sum(
        iono.arr["JH"][: len(iono.th) // 2, :]
        * iono.r_IS**2
        * np.sin(th[: len(iono.th) // 2, :])
        * daz[: len(iono.th) // 2, :]
        * dth[: len(iono.th) // 2, :]
    )
    iono.JH_S = np.sum(
        iono.arr["JH"][len(iono.th) // 2 : -1, :]
        * iono.r_IS**2
        * np.sin(th[len(iono.th) // 2 : -1, :])
        * daz[len(iono.th) // 2 :, :]
        * dth[len(iono.th) // 2 :, :]
    )


def calc_OCB(iono, sim, N_samp=100, r=4, disp=None):
    """Perform magnetic topology mapping on the simulation inner boundary.

    Performs magnetic topology mapping on the simulation inner boundary and extracts
    contours of domain edges.

    """
    import matplotlib.pyplot as plt
    from skimage import measure

    from ..magnetosphere.connectivity import calc_connectivity

    # Make sure magnetosphere and ionosphere are at same timestep
    if int(sim.time) != int(iono.time):
        sim.import_timestep(sim.timestep(iono.time))

    # MHD Inner boundary spherical coordinates on which to calculate connectivity
    du, dv = 2 * np.pi / N_samp, np.pi / N_samp
    us, vs = np.arange(0, 2 * np.pi + du, du), np.arange(0, np.pi + dv, dv)
    u, v = np.meshgrid(us, vs)  # Centre on North
    xsph = r * np.cos(u) * np.sin(v)
    ysph = r * np.sin(u) * np.sin(v)
    zsph = r * np.cos(v)

    # Rotating from gorgon frame into GSM frame -
    # rotate coordinates by dipole tilt angle
    xsph, ysph, zsph = (
        np.cos(-iono.tilt * np.pi / 180) * xsph
        + np.sin(-iono.tilt * np.pi / 180) * zsph,
        ysph,
        -np.sin(-iono.tilt * np.pi / 180) * xsph
        + np.cos(-iono.tilt * np.pi / 180) * zsph,
    )

    xlink = xsph.ravel()  # Unpacking mesh to calculate connectivity
    ylink = ysph.ravel()
    zlink = zsph.ravel()

    # Calculating connectivity
    link, _ = calc_connectivity(
        np.stack([xlink, ylink, zlink]).T,
        sim.arr["Bvec_c"],
        np.array([sim.dx[0], sim.dy[0], sim.dz[0]]),
        sim.center - 0.5 * sim.dx[0],
        ns=10000,
    )
    conn = link.reshape(xsph.shape)  # Same shape as mesh grid

    if disp == "2D":
        plt.pcolormesh(u * 180 / np.pi, v * 180 / np.pi, conn, zorder=2)
        plt.contour(u * 180 / np.pi, v * 180 / np.pi, conn, zorder=3, colors="w")
        plt.xlim(0, 360)
        plt.xticks(np.arange(0, 361, 30))
        plt.ylim(180, 0)
        plt.xlabel(r"Azimuth / $^\circ$")
        plt.ylabel(r"Colatitude / $^\circ$")
        plt.title(
            "Inner Boundary Connectivity at " + "%.2f" % r + r"$R_E$", fontsize=11
        )
        plt.savefig("IBConn", dpi=150)

    # Find OCB as edge of open field regions on inner boundary
    conn_open = np.where((conn != 3) & (conn != 4), 0, conn)
    contours = measure.find_contours(conn_open.T, 0)

    if disp == "3D":
        arr = conn

        import matplotlib as mpl

        cnorm = mpl.colors.Normalize(vmin=0, vmax=5)

        from matplotlib import cm

        cmap = "viridis"
        col1 = cm.ScalarMappable(cmap=cmap, norm=cnorm)

        # Creating contour plots on a 3D spherical grid

        fig1 = plt.figure(figsize=(15, 8), dpi=100)

        ax1 = fig1.add_subplot(121, projection="3d")
        ax1.plot_surface(
            xsph,
            ysph,
            zsph,
            cstride=1,
            rstride=1,
            facecolors=col1.to_rgba(arr),
            shade=False,
        )
        view_ang = (46, 225)
        ax1.view_init(elev=view_ang[0], azim=view_ang[1])
        ax1.set_xlabel(r"$X$ / $R_E$", fontsize=13)
        ax1.set_ylabel(r"$Y$ / $R_E$", fontsize=13)
        ax1.set_zlabel(r"$Z$ / $R_E$", fontsize=13)

        ax2 = fig1.add_subplot(122, projection="3d")
        ax2.plot_surface(
            xsph,
            ysph,
            zsph,
            cstride=1,
            rstride=1,
            facecolors=col1.to_rgba(arr),
            shade=False,
        )
        ax2.view_init(elev=-view_ang[0], azim=360 - view_ang[1])
        ax2.set_xlabel(r"$X$ / $R_E$", fontsize=13)
        ax2.set_ylabel(r"$Y$ / $R_E$", fontsize=13)
        ax2.set_zlabel(r"$Z$ / $R_E$", fontsize=13)

        plt.tight_layout()
        plt.show()

    OCBs_MS = []
    for con in contours:
        OCBs_MS.append(
            np.array([us[con[:, 0].astype(int)], vs[con[:, 1].astype(int)]]).T
        )

    # Dipole mapping down onto ionospheric grid (for plotting)
    OCBs_IS = []
    for OCB in OCBs_MS:
        OCB[:, 1] = np.where(
            OCB[:, 1] > np.pi / 2,
            np.pi - np.arcsin(np.sin(np.pi - OCB[:, 1]) * np.sqrt(r) ** -1),
            np.arcsin(np.sin(OCB[:, 1]) * np.sqrt(r) ** -1),
        )
        OCBs_IS.append(OCB)

    return OCBs_MS, OCBs_IS


def calc_FPC(sim, n_mask, disp=False):
    """Calculate total North/South-open magnetic flux.

    Calculate total North/South-open magnetic flux on outer boundaries of
    simulation domain.

    Args:
    ----
    sim (object): Simulation object.
    n_mask (int): Masking factor for gridpoints.
    disp (bool, optional): Whether to display plots. Defaults to False.

    Returns:
    -------
    tuple: Total North and South open magnetic fluxes.

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from ..magnetosphere.connectivity import calc_connectivity

    # Boundaries of box (planar grids)
    boundaries = {}
    boundaries["South"] = (
        np.meshgrid(sim.xc[::n_mask], sim.yc[::n_mask])[0],
        np.meshgrid(sim.xc[::n_mask], sim.yc[::n_mask])[1],
        np.zeros([len(sim.yc[::n_mask]), len(sim.xc[::n_mask])]) + sim.zc[::n_mask][0],
    )
    boundaries["North"] = (
        np.meshgrid(sim.xc[::n_mask], sim.yc[::n_mask])[0],
        np.meshgrid(sim.xc[::n_mask], sim.yc[::n_mask])[1],
        np.zeros([len(sim.yc[::n_mask]), len(sim.xc[::n_mask])]) + sim.zc[::n_mask][-1],
    )
    boundaries["Dusk"] = (
        np.meshgrid(sim.xc[::n_mask], sim.zc[::n_mask])[0],
        np.zeros([len(sim.zc[::n_mask]), len(sim.xc[::n_mask])]) + sim.yc[::n_mask][0],
        np.meshgrid(sim.xc[::n_mask], sim.zc[::n_mask])[1],
    )
    boundaries["Dawn"] = (
        np.meshgrid(sim.xc[::n_mask], sim.zc[::n_mask])[0],
        np.zeros([len(sim.zc[::n_mask]), len(sim.xc[::n_mask])]) + sim.yc[::n_mask][-1],
        np.meshgrid(sim.xc[::n_mask], sim.zc[::n_mask])[1],
    )
    boundaries["Downtail"] = (
        np.zeros([len(sim.zc[::n_mask]), len(sim.yc[::n_mask])]) + sim.xc[::n_mask][-1],
        np.meshgrid(sim.yc[::n_mask], sim.zc[::n_mask])[0],
        np.meshgrid(sim.yc[::n_mask], sim.zc[::n_mask])[1],
    )

    # If plotting, set colormap for connectivity arrays
    if disp:
        fig = plt.figure(figsize=(8, 30))
        i = 1
        cmap = plt.get_cmap("viridis")
        bounds = np.linspace(0, 5, 6)
        _ = mpl.colors.BoundaryNorm(bounds, cmap.N)

    B = sim.arr["Bvec_c"][::n_mask, ::n_mask, ::n_mask, :]

    fluxes = {}
    for boundary, (x, y, z) in boundaries.items():
        B_xy = B[:, :, sim.zc[::n_mask] == z[0, 0], :].squeeze()
        B_xz = B[:, sim.yc[::n_mask] == y[0, 0], :, :].squeeze()
        B_yz = B[sim.xc[::n_mask] == x[0, 0], :, :, :].squeeze()

        # Calculating connectivity
        xlink = x.ravel()
        ylink = y.ravel()
        zlink = z.ravel()
        link, _ = calc_connectivity(
            np.stack([xlink, ylink, zlink]).T,
            sim.arr["Bvec_c"],
            np.array([sim.dx[0], sim.dy[0], sim.dz[0]]),
            sim.center - 0.5 * sim.dx[0],
            ns=10000,
        )
        conn = link.reshape(x.shape)

        # Boolean arrays of gridpoints containing North/South-open flux
        N_ind = (conn == 4).T
        S_ind = (conn == 3).T

        # Total flux (will cumulatively sum as we iterate over each boundary)
        flux_N = 0
        flux_S = 0

        if disp:
            ax = fig.add_subplot(5, 1, i)
            i += 1

        # Calculating fluxes and (optionally) plotting
        if boundary == "South" or boundary == "North":
            if np.any(N_ind):
                B_N = B_xy[N_ind, :]
                flux_N = abs(np.sum(sim.dx[0] * sim.dy[0] * B_N[:, 2])) * n_mask**2
            if np.any(S_ind):
                B_S = B_xy[S_ind, :]
                flux_S = abs(np.sum(sim.dx[0] * sim.dy[0] * B_S[:, 2])) * n_mask**2
            if disp:
                _ = ax.imshow(
                    conn.T,
                    vmin=1,
                    vmax=np.max(bounds),
                    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                )
                ax.set_xlabel(r"$X$ / $R_E$")
                ax.set_ylabel(r"$Y$ / $R_E$")

        if boundary == "Dusk" or boundary == "Dawn":
            if np.any(N_ind):
                B_N = B_xz[N_ind, :]
                flux_N = abs(np.sum(sim.dx[0] * sim.dz[0] * B_N[:, 1])) * n_mask**2
            if np.any(S_ind):
                B_S = B_xz[S_ind, :]
                flux_S = abs(np.sum(sim.dx[0] * sim.dz[0] * B_S[:, 1])) * n_mask**2
            if disp:
                _ = ax.imshow(
                    conn.T,
                    vmin=1,
                    vmax=np.max(bounds),
                    extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                )
                ax.set_xlabel(r"$X$ / $R_E$")
                ax.set_ylabel(r"$Z$ / $R_E$")

        if boundary == "Downtail":
            if np.any(N_ind):
                B_N = B_yz[N_ind, :]
                flux_N = abs(np.sum(sim.dy[0] * sim.dz[0] * B_N[:, 0])) * n_mask**2
            if np.any(S_ind):
                B_S = B_yz[S_ind, :]
                flux_S = abs(np.sum(sim.dy[0] * sim.dz[0] * B_S[:, 0])) * n_mask**2
            if disp:
                _ = ax.imshow(
                    conn.T,
                    vmin=1,
                    vmax=np.max(bounds),
                    extent=[np.min(y), np.max(y), np.min(z), np.max(z)],
                )
                ax.set_xlabel(r"$Y$ / $R_E$")
                ax.set_ylabel(r"$Z$ / $R_E$")

        if disp:
            # cb = plt.colorbar(p, cmap=cmap, norm=norm, spacing='proportional',
            # ticks=bounds, boundaries=bounds, format='%i')
            ax.set_title(boundary)

        fluxes[boundary] = flux_N, flux_S

        tot_N, tot_S = 0, 0
        for boundary, (flux_N, flux_S) in fluxes.items():
            tot_N += flux_N
            tot_S += flux_S

    if disp:
        # plt.savefig(sim.time+'_Boundary_Fluxes')
        plt.show()

    return tot_N, tot_S


def calc_ExB_drift(iono, r=None, interp="linear"):
    """Calculate the ExB drift velocity for a the ionosphere object and radial distance.

    Args:
    ----
        iono (ionosphere object): The ionosphere object containing the necessary data.
        r (float, optional): The radial distance. Defaults to None.
        interp (str, optional): The interpolation method. Defaults to "linear".

    Returns:
    -------
        tuple: A tuple containing the E-field, B-field, ExB-drift velocity, and the
        x, y, z coordinates of the inner boundary.

    """
    if r is None:
        r = iono.r_MS
    rs = np.array([r - 0.3 * iono.r_IS, r, r + 0.3 * iono.r_IS])
    th_llbs = np.pi / 2 - np.arccos(np.sqrt(iono.r_IS / rs))
    phi_IBs = np.zeros([len(rs), len(iono.th), len(iono.az)])

    # Ionosphere and inner boundary coordinates
    th_IS, az_IS = np.swapaxes(np.meshgrid(iono.th, iono.az), 1, 2)
    x_IB = r * np.sin(th_IS) * np.cos(az_IS)
    y_IB = r * np.sin(th_IS) * np.sin(az_IS)
    z_IB = r * np.cos(th_IS)

    from scipy.interpolate import griddata

    for ir, th_llb in enumerate(th_llbs):
        for i in range(len(iono.th)):
            if iono.th[i] <= th_llb:
                edgeN = i
            if iono.th[i] >= np.pi - th_llb:
                edgeS = i
                break
        size = (edgeN + 1) * 2

        d = 1  # set to 1 to force phi=0 at equator for r=r_IB
        th_cut = np.zeros([size + d])
        phi_cut = np.zeros([size + d, len(iono.az)])
        for i in range(edgeN + 1):
            th_cut[i] = iono.th[i]
            for j in range(len(iono.az)):
                phi_cut[i, j] = iono.arr["phi"][i, j]
        for i in np.arange(edgeS, len(iono.th), 1):
            th_cut[i - (len(iono.th) - size) + d] = iono.th[i]
            for j in range(len(iono.az)):
                phi_cut[i - (len(iono.th) - size) + d, j] = iono.arr["phi"][i, j]

        th_IB = (
            -np.arccos(np.cos(-th_cut + np.pi / 2) * np.sqrt(rs[ir] / iono.r_IS))
            + np.pi / 2
        )  # dipole-mapping theta to r_ms
        if d > 0:
            th_IB[edgeN + 1] = np.pi / 2
            phi_cut[edgeN + 1, :] = 0.5 * (
                iono.arr["phi"][edgeN + 1, :] + iono.arr["phi"][edgeS - 1, :]
            )
        for i in range(len(th_IB)):
            if i > edgeN:
                th_IB[i] = np.pi - th_IB[i]
        # print(th_IB[:]*180/np.pi,phi_cut[0,:])

        # Interpolating from grid [th2,az2] back to full spherical grid [th1,az1]
        th_IB, az_IB = np.swapaxes(np.meshgrid(th_IB, iono.az), 1, 2)

        phi_IBs[ir, :, :] = griddata(
            (th_IB.ravel(), az_IB.ravel()),
            phi_cut.ravel(),
            (th_IS, az_IS),
            method=interp,
        )

    E = -np.array(
        gradient(rs, iono.th, iono.az, phi_IBs)
    )  # E-field in cartesian [3,th,ph]
    E[:, 0, :], E[:, -1, :] = E[:, 1, :], E[:, -2, :]  # Pole singularities

    Bx, By, Bz = calc_B_cart(x_IB, y_IB, z_IB)  # B-field unit vectors in cartesian
    B = np.array([Bx, By, Bz])
    Bmag = np.linalg.norm(B, axis=0)

    ExB = np.cross(E, B, 0, 0)  # Has dims [th,ph,3]
    ExB = np.swapaxes(np.swapaxes(ExB, 0, 1), 0, 2)  # Swapping to same dims as E,B
    v = np.divide(ExB, np.array([Bmag**2, Bmag**2, Bmag**2]))  # ExB-drift v = ExB/B^2

    return E, B, v, x_IB, y_IB, z_IB


def calc_B_cart(x, y, z, r_P=6.4e6, Beq=3.12e-5):
    """Calculate the Cartesian components of the magnetic field vector.

    Args:
    ----
        x (float): x-coordinate of the point in meters.
        y (float): y-coordinate of the point in meters.
        z (float): z-coordinate of the point in meters.
        r_P (float, optional): Radius of the planet in meters. Defaults to 6.4e6.
        Beq (float, optional): Equatorial magnetic field strength in Tesla.
        Defaults to 3.12e-5.

    Returns:
    -------
        tuple: A tuple containing the x, y, and z components of the magnetic field
        vector in Tesla.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arctan2(np.sqrt(x**2 + y**2), z)
    az = np.arctan2(y, x)

    Br = -2 * np.cos(th) * Beq * (r_P / r) ** 3
    Bth = -np.sin(th) * Beq * (r_P / r) ** 3

    Bx = np.sin(th) * np.cos(az) * Br + np.cos(th) * np.cos(az) * Bth
    By = np.sin(th) * np.sin(az) * Br + np.cos(th) * np.sin(az) * Bth
    Bz = np.cos(th) * Br - np.sin(th) * Bth

    return Bx, By, Bz


def gradient(r, th, ph, psi):
    """Calculate the gradient of a scalar potential on a spherical shell.

    For scalar potential psi on a spherical shell (const. r, but psi has r-dependence)
    Calculates numerical gradient using spherical coordinates
    Returns a 3-D vector field in cartesian coords
    """
    if np.size(r) == 1:  # generalise for a thin-shell (and remove radial dependence)
        r = np.array([r - 0.01 * r, r, r + 0.01 * r])
        psi = np.array([psi, psi, psi])
    gradpsi_r = 0 * psi
    gradpsi_th = 0 * psi  # r, th and ph components of the gradient
    gradpsi_ph = 0 * psi
    gradpsi_x = 0 * psi  # x, y and z components of the gradient
    gradpsi_y = 0 * psi
    gradpsi_z = 0 * psi

    # Iteratively calculate gradient using central differencing
    for i in range(len(th)):
        for j in range(len(ph)):
            dpsi_dr = (psi[2, i, j] - psi[0, i, j]) / (2 * (r[2] - r[0]))
            if i == 0:  # at the boundaries
                dpsi_dth = (psi[1, 1, j] - psi[1, 0, j]) / (th[1] - th[0])
            elif i == (len(th) - 1):
                dpsi_dth = (psi[1, len(th) - 1, j] - psi[1, len(th) - 2, j]) / (
                    th[len(th) - 1] - th[len(th) - 2]
                )
            else:  # everywhere else
                dpsi_dth = (psi[1, i + 1, j] - psi[1, i - 1, j]) / (
                    2 * (th[i + 1] - th[i])
                )
            if j == 0:  # at the boundaries
                dpsi_dph = (psi[1, i, 1] - psi[1, i, len(ph) - 1]) / (
                    2 * (ph[1] - ph[0])
                )
            elif j == (len(ph) - 1):
                dpsi_dph = (psi[1, i, 0] - psi[1, i, len(ph) - 2]) / (
                    2 * (ph[0] - ph[len(ph) - 1])
                )
            else:  # everywhere else
                dpsi_dph = (psi[1, i, j + 1] - psi[1, i, j - 1]) / (
                    2 * (ph[j + 1] - ph[j])
                )
            gradpsi_r[1, i, j] = 1 * dpsi_dr
            gradpsi_th[1, i, j] = 1 / r[1] * dpsi_dth
            if i == 0:  # singularity
                gradpsi_ph[1, i, j] = 0
            else:
                gradpsi_ph[1, i, j] = 1 / (r[1] * np.sin(th[i])) * dpsi_dph
            gradpsi_x[1, i, j] = (
                np.sin(th[i]) * np.cos(ph[j]) * gradpsi_r[1, i, j]
                + np.cos(th[i]) * np.cos(ph[j]) * gradpsi_th[1, i, j]
                - np.sin(ph[j]) * gradpsi_ph[1, i, j]
            )
            gradpsi_y[1, i, j] = (
                np.sin(th[i]) * np.sin(ph[j]) * gradpsi_r[1, i, j]
                + np.cos(th[i]) * np.sin(ph[j]) * gradpsi_th[1, i, j]
                + np.cos(ph[j]) * gradpsi_ph[1, i, j]
            )
            gradpsi_z[1, i, j] = (
                np.cos(th[i]) * gradpsi_r[1, i, j] - np.sin(th[i]) * gradpsi_th[1, i, j]
            )

    grad = [gradpsi_x[1, :, :], gradpsi_y[1, :, :], gradpsi_z[1, :, :]]

    return grad
