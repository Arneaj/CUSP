"""Module for the elecproject class."""
import datetime as dt

import numpy as np

from .coordinates import GEO_to_MAG


def Z_from_cond(cond_val, w):
    """Define the surface impedance Z.

    Helper function to define the surface impedance Z (in Ohms) for a homogenous Earth
    conductivity (in mS/m). Assumption is that Z=mu0E/B, and in turn the skin depth
    p=Z/(mu0iw).

    Args:
    ----
        cond_val (float): Conductivity in mS/m
        w (numpy array): Angular frequencies to be used

    """
    return np.sqrt((1j * w * 4 * np.pi * 1e-7) / (cond_val * 1e-3))


class elecproject:
    """Class for electrojet projection."""

    def __init__(self, station, height=110e3, size=1000e3):
        """Initialise the elecproject class.

        Elecproject class on which subsets of the ionospheric horizontal current
        are taken for computing the ground electric and magnetic field at a given
        location on the Earth. This location should be within close proximity of the
        auroral electrojet in order for the method to be sufficiently accurate.

        Args:
        ----
            station (str): Name of magnetometer station. Can be given in IAGA format or
            as named on INTERMAGNET. Otherwise requires list with station name and
            geographic colatitude, longitude in radians, e.g. ['ESK', 0.605, 6.227].
            height (float, optional): Height of electrojet above ground.
            Defaults to 100e3.
            size ([type], optional): Size of subset of ionosphere in metres.
            Defaults to 1000e3.

        """
        self.height = height
        self.radius = 6.371e6 + self.height  # Radius from centre of Earth
        self.size = size
        if isinstance(station, str):
            self.station = station
            from .coordinates import get_station_coords

            self.loc = get_station_coords(self.station)
        else:
            self.station = station[0]
            self.loc = station[1:]

    def sample_region(self, timeseries, sim_coords=False):
        """Extract ionospheric data within the electrojet region.

        Extracts a subset of the ionospheric data within the defined electrojet region
        over a given time range.

        Loads the global ionospheric data from an ionosphere timeseries dictionary.
        A subset of the data is taken directly within the approximate region, and the
        corresponding range of longitudes, latitudes and electric potential are stored
        for later plotting.

        For accurate analysis purposes, a sterographic projection of the ionosphere is
        generated and the potential is sampled onto a rectangular grid of radius
        self.size x self.size. The electric field is then calculated as the gradient of
        this, and used to compute the horizontal ionospheric current within the
        specified electrojet region. This is stored for later analysis.

        Args:
        ----
            timeseries (dict): Ionospheric timeseries dictionary generated using the
            import_timerange function in the ionosphere module.
            sim_coords (str, optional): Choose whether to sample on simulation
            coordinates rather than 'GEO' or 'MAG'. Defaults to False.

        """
        # Load global ionospheric timeseries in specified coordinate system
        self.times = timeseries["times"]
        if "datetimes" in timeseries.keys():
            self.datetimes = timeseries["datetimes"]
            starttime_UT = self.datetimes[0]
        else:
            self.datetimes = None
            starttime_UT = dt.datetime.now()

        arrs_global = {}
        if not sim_coords:
            self.coords = timeseries["coords"]
            for var in timeseries["arr_names"]:
                arrs_global[var] = timeseries[var + "_" + self.coords].copy()
        else:
            for var in timeseries["arr_names"]:
                arrs_global[var] = timeseries[var].copy()
            self.coords = "sim"

        # Getting ionospheric data in sample region (in sph coords) for later plotting
        sigma = self.size / self.radius  # Angular distance (in latitude)
        if self.coords == "MAG":
            lat_stn, lon_stn = GEO_to_MAG(
                np.pi / 2 - self.loc[0], self.loc[1], starttime_UT
            )
            clt_stn = np.pi / 2 - lat_stn
        else:
            clt_stn, lon_stn = self.loc[0], self.loc[1]
        th_min, th_max = clt_stn - sigma, clt_stn + sigma
        az_min, az_max = (
            lon_stn - sigma,
            lon_stn + sigma,
        )  # Just pick same range in longitude
        th_inds = (timeseries["th"] > th_min) & (
            timeseries["th"] < th_max
        )  # Find indices within spatial range

        def az_norm(az, az_0):
            # Account for [2*pi,0] longitude boundary
            return np.remainder(az + (np.pi - az_0), 2 * np.pi)

        az_inds = (az_norm(timeseries["az"], lon_stn) > az_norm(az_min, lon_stn)) & (
            az_norm(timeseries["az"], lon_stn) < az_norm(az_max, lon_stn)
        )
        self.iono_sample = {
            "th": timeseries["th"][th_inds],
            "az": timeseries["az"][az_inds],
        }  # Store coordinates range for later
        for var in timeseries["arr_names"]:  # Sample global arrays within range
            self.iono_sample[var] = arrs_global[var][th_inds, :, :][:, az_inds, :]

        # Grid of spherical polar coords, such that sample point is at zero longitude
        # (so can have X pointing north)
        azim, colat = np.meshgrid(timeseries["az"] - lon_stn - np.pi, timeseries["th"])

        # Stereographic projection onto 2-D plane
        # (distortion is minor in auroral regions)
        x = np.sin(colat) * np.cos(azim)
        y = np.sin(colat) * np.sin(azim)
        z = np.cos(colat)
        X, Y = self.radius * np.array(
            [
                2 * x / (np.where(1 + z == 0, 1e10, 1 + z)),
                2 * y / np.where(1 + z == 0, 1e10, 1 + z),
            ]
        )
        X = np.where(z > 0, X, -X)  # correction for southern hemisphere

        # ... and for coords of sample point
        th_0, az_0 = clt_stn, -np.pi
        x_0 = np.sin(th_0) * np.cos(az_0)
        y_0 = np.sin(th_0) * np.sin(az_0)
        z_0 = np.cos(th_0)
        X_0, Y_0 = (
            self.radius * 2 * x_0 / (1 + np.abs(z_0)),
            self.radius * 2 * y_0 / (1 + np.abs(z_0)),
        )
        X_0 = np.where(z_0 > 0, X_0, -X_0)  # correction for southern hemisphere

        # Setting origin to sample point
        X, Y = X - X_0, Y - Y_0

        # Constructing rectangular grid about sample point
        self.d_cell = 100e3  # Grid cell size
        X_rec = np.arange(-self.size, self.size + 1, self.d_cell)
        Y_rec = 1 * X_rec

        # Interpolating potential and conductances onto rectangular grid
        self.X, self.Y = np.meshgrid(X_rec, Y_rec, indexing="ij")
        self.arr = {}
        self.arr["phi"] = np.zeros([len(X_rec), len(Y_rec), len(self.times)])
        from scipy import interpolate

        for var in timeseries["arr_names"]:
            self.arr[var] = 0 * self.arr["phi"]
            for i in range(
                len(self.times)
            ):  # Note correction to sign of Y.ravel() to ensure Eastward y-component
                self.arr[var][:, :, i] = interpolate.griddata(
                    np.array([X.ravel(), -Y.ravel()]).T,
                    arrs_global[var][:, :, i].ravel(),
                    (self.X, self.Y),
                    method="cubic",
                )  # ,fill_value=0)

        # E = -grad(phi)
        Ex, Ey = np.gradient(-self.arr["phi"], self.d_cell, axis=(0, 1))
        self.arr["E"] = np.array([Ex, Ey])

        # Electrojet: j_perp = sig_P*E + sig_H*(b x E)...
        # ... where b = (0,0,1) in North, so b x E = (-Ey, Ex) in North and (Ey, -Ex) in
        # South given NEZ co-ordinates
        if th_0 <= np.pi / 2:
            bz = 1
        else:
            bz = -1
        self.arr["j_perp"] = np.array(
            [self.arr["sig_P"] * Ex, self.arr["sig_P"] * Ey]
        ) + np.array([-bz * self.arr["sig_H"] * Ey, bz * self.arr["sig_H"] * Ex])
        self.arr["j_perp_hall"] = np.array(
            [-bz * self.arr["sig_H"] * Ey, bz * self.arr["sig_H"] * Ex]
        )
        self.arr["j_perp_pedersen"] = np.array(
            [self.arr["sig_P"] * Ex, self.arr["sig_P"] * Ey]
        )
        self.arr_names = np.append(
            timeseries["arr_names"], ["E", "j_perp", "j_perp_hall", "j_perp_pedersen"]
        )

    def calc_ground_fields(
        self,
        Z=0.0011 + 0.0015j,
        t_window=30 * 60,
        ignore_FAC=False,
        conduct="both",
        debug=False,
        cond=False,
        cond_val=1,
    ):
        """Generate ground electric and magnetic field timeseries.

        These are calculated using the complex image method (CIM) using horizontal
        currents, adapted from the method of Pulkkinen (2007). U-shaped current
        filaments are created at each rectangular grid point, and CIM is performed
        independently at each of these points to get the contribution to the horizontal
        electric and magnetic fields at the specified coordinates. Each contribution is
        then summed to get the net fields.

        Args:
        ----
            Z (complex, optional): Complex impedence of the solid Earth, assuming
            Z=mu0E/B, i.e. surface impedance in Ohms. Defaults to 0.0011+0.0015j Ohm for
            a single value, but realistically Z is frequency dependent
            (even in the homogenous Earth case).
            t_window (int, optional): FFT sampling window in seconds. Defaults to 1800.
            conduct (str, optional): Choose which source of ionospheric conductivity to
            use (hall,pedersen or both, defaults to 'both')
            ignore_FAC (bool, optional): Optionally ignore the field-aligned current
            contribution to the ground field.
            debug (bool, optional): If True, will plot the power spectrum of each
            U-shaped current filament. Defaults to False.
            cond (bool, optional): If True, Z is calculated from homogenous
            conductivity, with the conductivity input passed in as Z. Defaults to False
            cond_val (float, optional): If cond=True, value of conductivity to use in
            mS/m. Defaults to 1 mS/m.

        """
        # Length of each U-shaped current filament (width of rectangular grid cell)
        Lx = self.X[1, 0] - self.X[0, 0]
        Ly = self.Y[0, 1] - self.Y[0, 0]

        # Initialise ground electric and magnetic field arrays
        self.E_ground = np.zeros([3, self.times[-1] - self.times[0] + t_window // 2])
        self.B_ground = 0 * self.E_ground

        from .CIM import CIM_calc, get_spectrum, get_timeseries

        # Iterate through the grid and calculate contributions to net field at location
        # on ground
        for i in range(len(self.X[:, 0]) - 1):
            for j in range(len(self.X[0, :]) - 1):
                # Dummy indices forced inside grid edges (we will need to iterate
                # outside just the grid)
                ic, jc = i, j
                ir, ju = i + 1, j + 1

                # Define station coords wrt the U-current filaments (at cell, cell to
                # right and cell above)
                # Requires having the current filament at the origin i.e. r_station -
                # r_current = (0,0,0) - r_current
                x, x_r, x_u = (
                    -self.X[ic, jc],
                    -self.X[ir, jc],
                    -self.X[ic, ju],
                )  # just make negative as station is at grid origin
                y, y_r, y_u = -self.Y[ic, jc], -self.Y[ir, jc], -self.Y[ic, ju]
                z, _, _ = 0, 0, 0

                # At the grid edges let currents be carried to/from 'infinity'
                # (very far away)
                if i == 0:  # Left edge
                    x = 1e10 * Lx
                    x_u = 1e10 * Lx
                elif i == len(self.X[:, 0]) - 2:  # Right edge
                    x_r = -1e10 * Lx
                if j == 0:  # Bottom edge
                    y = 1e10 * Ly
                    y_r = 1e10 * Ly
                elif j == len(self.X[0, :]) - 2:  # Top edge
                    y_u = -1e10 * Ly

                # Now we define new cartesian coords for CIM: y parallel to electrojet,
                # x perpendicular, z into the Earth
                # On our grid we have x pointing North (geographic or geomagnetic), y
                # pointing East and z into the Earth
                # For an x-directed current element we thus do the transformation
                # (x, y, z) -> (y, -x, z) and Ix -> - Ix
                # For a y-directed current element we require no transformation

                # Cylindrical coords of station, measured from FAC filament on cell of
                # interest
                rho_r_1 = np.sqrt(y**2 + x**2)
                ph_r_1 = np.arctan2(x, y)

                # Cylindrical coords of station, measured from FAC filament on cell to
                # the right
                rho_r_2 = np.sqrt(y_r**2 + x_r**2)
                ph_r_2 = np.arctan2(x_r, y_r)

                # Cylindrical coords of station, measured from FAC filament on cell of
                # interest
                rho_u_1 = np.sqrt(y**2 + x**2)
                ph_u_1 = np.arctan2(y, x)

                # Cylindrical coords of station, measured from FAC filament on cell
                # upwards
                rho_u_2 = np.sqrt(y_u**2 + x_u**2)
                ph_u_2 = np.arctan2(y_u, x_u)

                # Amplitudes and frequencies of harmonics of X-directed U-current
                # filament calculated from an STFT...
                if conduct == "both":
                    I_x = (
                        (
                            self.arr["j_perp"][0, ic, jc, :]
                            + self.arr["j_perp"][0, ir, jc, :]
                        )
                        / 2
                        * Ly
                    )
                elif conduct == "hall":
                    I_x = (
                        (
                            self.arr["j_perp_hall"][0, ic, jc, :]
                            + self.arr["j_perp_hall"][0, ir, jc, :]
                        )
                        / 2
                        * Ly
                    )
                elif conduct == "pedersen":
                    I_x = (
                        (
                            self.arr["j_perp_pedersen"][0, ic, jc, :]
                            + self.arr["j_perp_pedersen"][0, ir, jc, :]
                        )
                        / 2
                        * Ly
                    )

                t, f_x, I_x, phase_x = get_spectrum(
                    self.times, I_x, t_window=t_window, disp=debug
                )

                if cond:
                    Z = Z_from_cond(cond_val, 2 * np.pi * f_x)

                # ... and resulting E and B fields from CIM as time series via inverse
                # STFT
                E_x, B_x = CIM_calc(
                    Lx,
                    self.height,
                    Z,
                    -I_x,
                    phase_x,
                    2 * np.pi * f_x,
                    y,
                    (x + x_r) / 2,
                    z,
                    rho_r_1,
                    ph_r_1,
                    rho_r_2,
                    ph_r_2,
                    ignore_FAC,
                )
                E_x = np.array([E_x[1, :, :], E_x[0, :, :], E_x[2, :, :]])
                B_x = np.array([B_x[1, :, :], B_x[0, :, :], B_x[2, :, :]])

                self.mag_times, E_x, B_x = get_timeseries(E_x, B_x, t_window)

                # Amplitudes and frequencies of harmonics of Y-directed U-current
                # filament calculated from an STFT...
                if conduct == "both":
                    I_y = (
                        (
                            self.arr["j_perp"][1, ic, jc, :]
                            + self.arr["j_perp"][1, ic, ju, :]
                        )
                        / 2
                        * Lx
                    )
                elif conduct == "hall":
                    I_y = (
                        (
                            self.arr["j_perp_hall"][1, ic, jc, :]
                            + self.arr["j_perp_hall"][1, ic, ju, :]
                        )
                        / 2
                        * Lx
                    )
                elif conduct == "pedersen":
                    I_y = (
                        (
                            self.arr["j_perp_pedersen"][1, ic, jc, :]
                            + self.arr["j_perp_pedersen"][1, ic, ju, :]
                        )
                        / 2
                        * Lx
                    )

                _, f_y, I_y, phase_y = get_spectrum(
                    self.times, I_y, t_window=t_window, disp=debug
                )

                if cond:
                    Z = Z_from_cond(cond_val, 2 * np.pi * f_y)

                # ... and resulting E and B fields from CIM as time series via inverse
                # STFT
                E_y, B_y = CIM_calc(
                    Ly,
                    self.height,
                    Z,
                    I_y,
                    phase_y,
                    2 * np.pi * f_y,
                    x,
                    (y + y_u) / 2,
                    z,
                    rho_u_1,
                    ph_u_1,
                    rho_u_2,
                    ph_u_2,
                    ignore_FAC,
                )
                E_y = np.array([E_y[0, :, :], E_y[1, :, :], E_y[2, :, :]])
                B_y = np.array([B_y[0, :, :], B_y[1, :, :], B_y[2, :, :]])

                _, E_y, B_y = get_timeseries(E_y, B_y, t_window)

                # Finally, sum up all the contributions across the grid!
                self.E_ground += E_x + E_y
                self.B_ground += B_x + B_y
        self.mag_times, self.E_ground, self.B_ground = (
            self.mag_times[: -int(t_window // 2) + 1] + self.times[0],
            self.E_ground[:, : -int(t_window // 2) + 1],
            self.B_ground[:, : -int(t_window // 2) + 1],
        )
