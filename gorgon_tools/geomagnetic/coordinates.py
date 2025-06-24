"""Functions for transforming between different geomagnetic coordinate systems."""
import datetime as dt

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


def get_station_coords(station_name):
    """Return the colatitude, longitude of a given magnetometer station on INTERMAGNET.

    Args:
    ----
        station_name (str): Name of magnetometer station. Can be given in IAGA format or
        as named on INTERMAGNET.

    Returns:
    -------
        (float, float): Colatitude and longitude of station in radians.

    """
    from io import BytesIO
    from pkgutil import get_data

    import pandas as pd

    data = get_data(__name__, "data/supermag-stations.csv")
    stations = pd.read_csv(BytesIO(data), encoding="utf8")

    if np.any(stations[stations["STATION-NAME"] == station_name]):
        station = stations[stations["STATION-NAME"] == station_name]
        th, az = 90 - float(station["GEOLAT"]), float(station["GEOLON"])
    elif np.any(stations[stations["IAGA"] == station_name]):
        station = stations[stations["IAGA"] == station_name]
        th, az = 90 - float(station["GEOLAT"]), float(station["GEOLON"])
    else:
        AE_syn = ["A" + ("%02d" % i) for i in range(0, 48)]
        KP_syn = ["K" + ("%02d" % i) for i in range(0, 48)]
        grid_syn = ["G" + "%03d" % i for i in range(0, 1000)]

        if station_name in AE_syn:
            clt = 17.5 if int(station_name[-2:]) < 24 else 180 - 17.5
            th, az = clt, ((int(station_name[-2:])) % 24) * 360 / 24
        elif station_name in KP_syn:
            clt = 30 if int(station_name[-2:]) < 24 else 180 - 30
            th, az = clt, ((int(station_name[-2:])) % 24) * 360 / 24
        elif station_name in grid_syn:
            clts = np.append(np.arange(5, 41, 5), 180 - np.arange(5, 41, 5))
            lons = np.arange(0, 360, 6)
            thg, azg = np.meshgrid(clts, lons)
            th, az = (
                thg.ravel()[int(station_name[-3:])],
                azg.ravel()[int(station_name[-3:])],
            )
        else:
            raise RuntimeWarning("Station '" + str(station_name) + "' not found.")
            th, az = 0, 0

    return th * np.pi / 180, az * np.pi / 180


def calc_dipole_axis(time, coords="GSE"):
    """Calculate the dipole axis unit vector.

    Calculates the dipole axis unit vector in a chosen coordinate system using IGRF data
    (backdated to 1590 AD) from http://wdc.kugi.kyoto-u.ac.jp/poles/polesexp.html
    (as of 19/10/2021) to obtain geomagnetic pole coordinates.

    More information at https://www.ngdc.noaa.gov/geomag/GeomagneticPoles.shtml.

    Args:
    ----
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to calculate dipole axis unit vector.
        coords (str, optional): Coordinate system for the calculated unit vector,
        choices are 'GEO', 'GEI', 'GSE', or 'GSM'. Defaults to 'GSE'.

    Returns:
    -------
        (np.array, np.array, np.array): Cartesian components of the dipole axis unit
        vector (Mx, My, Mz)

    """
    from .igrf import IGRF_load_coeffs

    g, h = IGRF_load_coeffs(time, 1)
    M = np.sqrt(g[0, 0] ** 2 + g[0, 1] ** 2 + h[0, 1] ** 2)
    M_lat = np.arccos(-g[0, 0] / M)
    M_lon = np.arcsin(-h[0, 1] / np.sqrt(g[0, 1] ** 2 + h[0, 1] ** 2))

    M_x, M_y, M_z = polar_to_cart(np.pi / 2 - M_lat, M_lon)

    if coords == "GEO":
        return M_x, M_y, M_z
    elif coords in ["GEI", "GSE", "GSM", "SM", "SMD"]:
        M_x, M_y, M_z = GEO_to_GEI(M_x, M_y, M_z, time)
        if coords == "GEI":
            return M_x, M_y, M_z
        else:
            M_x, M_y, M_z = GEI_to_GSE(M_x, M_y, M_z, time)
            if coords == "GSE":
                return M_x, M_y, M_z
            else:
                M_x, M_y, M_z = GSE_to_GSM(M_x, M_y, M_z, time)
                if coords == "GSM":
                    return M_x, M_y, M_z
                else:
                    M_x, M_y, M_z = GSM_to_SM(M_x, M_y, M_z, time)
                    if coords == "SM":
                        return M_x, M_y, M_z
                    else:
                        M_x, M_y, M_z = SM_to_SMD(M_x, M_y, M_z, time)
                        return M_x, M_y, M_z
    else:
        print("Please pick a valid coordinate system.")


def GEO_to_MAG(lat, lon, time, inv=False):
    """Transform coordinates between geographic and geomagnetic spherical polar systems.

    Transforms geographic (GEO) spherical polar coordinates to geomagnetic (MAG)
    spherical polar coordinates, following definition of Hapgood (1992). Can optionally
    perform inverse transformation.

    Accounts for geomagnetic timscale changes (backdated to 1590 AD) in dipole axis
    location to within 0.2 degrees - tested against
    http://wdc.kugi.kyoto-u.ac.jp/igrf/gggm/index.html (as of 19/10/2021).

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        lat (np.array): Geographic latitude in radians,
        or geomagnetic latitude if inv = True.
        lon (np.array): Geographic longitude in radians,
        or geomagnetic longitude if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (MAG to GEO).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array): Resulting geomagnetic spherical polar coordinates
        (latitude, longitude), or geographic equivalent if inv = True.

    """
    M_x, M_y, M_z = calc_dipole_axis(time, coords="GEO")
    M_lat, M_lon = cart_to_polar(M_x, M_y, M_z)

    # Allow flexibility for vx, vy, vz array dimensions
    if (type(M_x) is np.ndarray and np.size(time) > 1) and M_x.shape != np.array(
        time
    ).shape:
        M_lat = np.tile(M_lat, np.append(M_x.shape[:-1], [1]))
        M_lon = np.tile(M_lon, np.append(M_x.shape[:-1], [1]))
    # elif np.size(time) == 1:
    #     M_lat, M_lon = M_lat[0], M_lon[0]

    X, Y, Z = polar_to_cart(lat, lon)
    if not inv:
        X, Y, Z = rot_z(X, Y, Z, M_lon)
        X, Y, Z = rot_y(X, Y, Z, M_lat - np.pi / 2)
    else:
        X, Y, Z = rot_y(X, Y, Z, np.pi / 2 - M_lat)
        X, Y, Z = rot_z(X, Y, Z, -M_lon)

    lat, lon = cart_to_polar(X, Y, Z)

    return lat, lon


def GEO_to_GEI(v_x, v_y, v_z, time, inv=False):
    """Transform coordinates between geographic and geocentric equatorial inertial.

    Transforms a cartesian vector from geographic (GEO) coordinates to geocentric
    equatorial inertial (GEI) coordinates, following definition of Hapgood (1992). Can
    optionally perform inverse transformation.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in GEO coordinates,
        or GEI equivalent if inv = True.
        v_y (np.array): Y-component of vector in GEO coordinates,
        or GEI equivalent if inv = True.
        v_z (np.array): Z-component of vector in GEO coordinates,
        or GEI equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (GEI to GEO).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting GEI vector components (v_x, v_y, v_z),
        or GEO equivalent if inv = True.

    """
    MJD = (np.array(time) - dt.datetime(1858, 11, 17)).astype("timedelta64[D]")
    T0 = (MJD.astype(float) - 51544.5) / 36525.0
    UT = (
        pd.to_datetime(time) - pd.to_datetime((np.array(time)).astype("datetime64[D]"))
    ).astype("timedelta64[s]").total_seconds() / 3600
    theta = (100.461 + 36000.770 * T0 + 15.04107 * UT) * np.pi / 180

    # Allow flexibility for vx, vy, vz array dimensions
    if (type(v_x) is np.ndarray and np.size(time) > 1) and v_x.shape != np.array(
        time
    ).shape:
        theta = np.tile(theta, np.append(v_x.shape[:-1], [1]))
    elif np.size(time) == 1:
        theta = theta[0]

    if not inv:
        v_x, v_y, v_z = rot_z(v_x, v_y, v_z, -theta)
    else:
        v_x, v_y, v_z = rot_z(v_x, v_y, v_z, theta)

    return v_x, v_y, v_z


def GEI_to_GSE(v_x, v_y, v_z, time, inv=False):
    """Transform coordinates between GEI and geocentric solar ecliptic systems.

    Transforms a cartesian vector from geocentric equatorial interial (GEI) coordinates
    to geocentric solar ecliptic (GSE) coordinates, following definition of Hapgood
    (1992). Can optionally perform inverse transformation.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in GEI coordinates,
        or GSE equivalent if inv = True.
        v_y (np.array): Y-component of vector in GEI coordinates,
        or GSE equivalent if inv = True.
        v_z (np.array): Z-component of vector in GEI coordinates,
        or GSE equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (GSE to GEI).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting GSE vector components (v_x, v_y, v_z),
        or GEI equivalent if inv = True.

    """
    MJD = (np.array(time) - dt.datetime(1858, 11, 17)).astype("timedelta64[D]")
    T0 = (MJD.astype(float) - 51544.5) / 36525.0
    UT = (
        pd.to_datetime(time) - pd.to_datetime((np.array(time)).astype("datetime64[D]"))
    ).astype("timedelta64[s]").total_seconds() / 3600
    epsilon = (23.439 - 0.013 * T0) * np.pi / 180
    M = (357.528 + 35999.050 * T0 + 0.04107 * UT) * np.pi / 180
    L = (280.460 + 36000.772 * T0 + 0.04107 * UT) * np.pi / 180
    lambd = (
        L + ((1.915 - 0.0048 * T0) * np.sin(M) + 0.020 * np.sin(2 * M)) * np.pi / 180
    )

    # Allow flexibility for vx, vy, vz array dimensions
    if (type(v_x) is np.ndarray and np.size(time) > 1) and v_x.shape != np.array(
        time
    ).shape:
        epsilon = np.tile(epsilon, np.append(v_x.shape[:-1], [1]))
        lambd = np.tile(lambd, np.append(v_x.shape[:-1], [1]))
    elif np.size(time) == 1:
        epsilon, lambd = epsilon[0], lambd[0]

    if not inv:
        v_x, v_y, v_z = rot_x(v_x, v_y, v_z, epsilon)
        v_x, v_y, v_z = rot_z(v_x, v_y, v_z, lambd)
    else:
        v_x, v_y, v_z = rot_z(v_x, v_y, v_z, -lambd)
        v_x, v_y, v_z = rot_x(v_x, v_y, v_z, -epsilon)

    return v_x, v_y, v_z


def GSE_to_GSM(v_x, v_y, v_z, time, inv=False):
    """Transform coordinates between GSE and GSM systems.

    Transforms a cartesian vector from geocentric solar ecliptic (GSE) coordinates to
    geocentric solar magnetic (GSM) coordinates, following definition of Hapgood (1992).
    Can optionally perform inverse transformation.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in GSE coordinates,
        or GSM equivalent if inv = True.
        v_y (np.array): Y-component of vector in GSE coordinates,
        or GSM equivalent if inv = True.
        v_z (np.array): Z-component of vector in GSE coordinates,
        or GSM equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (GSM to GSE).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting GSM vector components (v_x, v_y, v_z),
        or GSE equivalent if inv = True.

    """
    _, M_y, M_z = calc_dipole_axis(time, "GSE")
    psi = np.arctan(M_y / M_z)

    # Allow flexibility for vx, vy, vz array dimensions
    if (type(v_x) is np.ndarray and np.size(time) > 1) and v_x.shape != np.array(
        time
    ).shape:
        psi = np.tile(psi, np.append(v_x.shape[:-1], [1]))

    if not inv:
        v_x, v_y, v_z = rot_x(v_x, v_y, v_z, -psi)
    else:
        v_x, v_y, v_z = rot_x(v_x, v_y, v_z, psi)

    return v_x, v_y, v_z


def GSE_to_SM(v_x, v_y, v_z, time, inv=False):
    """Transform coordinates between GSE and SM systems.

    Transforms a cartesian vector from geocentric solar ecliptic (GSE) coordinates to
    solar magnetic (SM) coordinates, following definition of Hapgood (1992). Can
    optionally perform inverse transformation.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in GSE coordinates,
        or SM equivalent if inv = True.
        v_y (np.array): Y-component of vector in GSE coordinates,
        or SM equivalent if inv = True.
        v_z (np.array): Z-component of vector in GSE coordinates,
        or SM equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (SM to GSE).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array):  Resulting SM vector components (v_x, v_y, v_z),
        or GSE equivalent if inv = True.

    """
    if not inv:
        v_x, v_y, v_z = GSE_to_GSM(v_x, v_y, v_z, time)
        v_x, v_y, v_z = GSM_to_SM(v_x, v_y, v_z, time)
    else:
        v_x, v_y, v_z = GSM_to_SM(v_x, v_y, v_z, time, inv)
        v_x, v_y, v_z = GSE_to_GSM(v_x, v_y, v_z, time, inv)

    return v_x, v_y, v_z


def GSM_to_SM(v_x, v_y, v_z, time, inv=False):
    """Transform coordinates between GSM and SM systems.

    Transforms a cartesian vector from geocentric solar magnetic (GSM) coordinates to
    solar magnetic (SM) coordinates, following definition of Hapgood (1992).
    Can optionally perform inverse transformation.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in GSM coordinates,
        or SM equivalent if inv = True.
        v_y (np.array): Y-component of vector in GSM coordinates,
        or SM equivalent if inv = True.
        v_z (np.array): Z-component of vector in GSM coordinates,
        or SM equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (SM to GSM).
        Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting SM vector components (v_x, v_y, v_z),
        or GSM equivalent if inv = True.

    """
    M_x, M_y, M_z = calc_dipole_axis(time, coords="GSM")
    mu = np.arctan(M_x / np.sqrt(M_y**2 + M_z**2))

    # Allow flexibility for vx, vy, vz array dimensions
    if (type(v_x) is np.ndarray and np.size(time) > 1) and v_x.shape != np.array(
        time
    ).shape:
        mu = np.tile(mu, np.append(v_x.shape[:-1], [1]).astype(int))

    if not inv:
        v_x, v_y, v_z = rot_y(v_x, v_y, v_z, -mu)
    else:
        v_x, v_y, v_z = rot_y(v_x, v_y, v_z, mu)

    return v_x, v_y, v_z


def SM_to_SMD(v_x, v_y, v_z, time, inv=False, gorgonops=False):
    """Transform coordinates between SM and SMD systems.

    Transforms a cartesian vector from geocentric solar magnetic (GSM) coordinates to
    diurnally averaged solar magnetic (SM) coordinates, called SMD here, which have the
    Z-axis aligned with the average dipole axis for a given day (i.e. time of year).

    This has the benefit of reducing the angle to the Sun-Earth line during periods near
    solstice, and hence reduces the inflow angle of the solar wind versus SM
    coordinates. Specifically, this will never be more than about +/-10 degrees, versus
    +/-33 degrees in SM. The downside is that the mean tilt angle must be taken as
    constant over the given period, rather than a rolling average. This allows for, and
    indeed requires, a unique dipole tilt angle for each Gorgon simulation.

    If a single datetime is provided, or the time range is less than 24 hours, the mean
    dipole tilt angle for that calendar day (or 2 calendar days) will be calculated.
    If the time range is greater than 24 hours then the mean for the day in the middle
    of the time range will be used; the range should span no more than a few days in
    order to avoid noticeable seasonal changes.

    Accepts 1-D arrays, or n-D arrays of vector components where the final axis has the
    same size as the time array.

    Args:
    ----
        v_x (np.array): X-component of vector in SM coordinates,
        or SMD equivalent if inv = True.
        v_y (np.array): Y-component of vector in SM coordinates,
        or SMD equivalent if inv = True.
        v_z (np.array): Z-component of vector in SM coordinates,
        or SMD equivalent if inv = True.
        time (np.array or datetime): Array of datetimes (or single datetime) in UT at
        which to perform transformation.
        inv (bool, optional): Option to perform inverse transformation (SMD to SM).
        Defaults to False.
        gorgonops (boolean, optional): Option to constrain tilt angle to a multiple of 5
        degree for operational applications. Defaults to False.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting SMD vector components (v_x, v_y, v_z),
        or SM equivalent if inv = True.

    """
    M_x, M_y, M_z = calc_dipole_axis(time, coords="GSM")
    mu = np.arctan(M_x / np.sqrt(M_y**2 + M_z**2))

    if np.size(time) > 1:
        if (
            pd.to_datetime(time[-1]) - pd.to_datetime(time[0])
        ).total_seconds() >= 3600 * 24:
            mu_avg = daily_avg_mu(time[len(time) // 2])
        else:
            mu_1 = daily_avg_mu(pd.to_datetime(time[0]))
            mu_2 = daily_avg_mu(pd.to_datetime(time[-1]))
            days = time.astype("datetime64[D]")
            if days[0] == days[-1]:
                mu_avg = mu_1
            else:
                n_day1, n_day2 = np.sum(days == days[0]), np.sum(days == days[-1])
                mu_avg = (mu_1 * n_day1 + mu_2 * n_day2) / len(time)
    elif type(time) is np.ndarray:
        mu_avg = daily_avg_mu(time[0])
    else:
        mu_avg = daily_avg_mu(time)

    if gorgonops:
        mu_avg = (5.0 * np.pi / 180.0) * round(mu_avg / (5.0 * np.pi / 180.0))
    del_mu = mu - mu_avg

    # Allow flexibility for vx, vy, vz array dimensions
    if type(v_x) is np.ndarray and np.size(time) > 1:
        mu = np.tile(mu_avg, np.append(v_x.shape[:-1], [1]))
    else:
        mu = mu_avg

    if not inv:
        v_x, v_y, v_z = rot_y(v_x, v_y, v_z, mu)
    else:
        v_x, v_y, v_z = rot_y(v_x, v_y, v_z, -mu)

    return v_x, v_y, v_z, mu_avg * 180 / np.pi, del_mu

    return (
        v_x,
        v_y,
        v_z,
        mu_avg * 180 / np.pi,
        del_mu,
    )  # note at the average dipole tilt here is in degrees


def daily_avg_mu(time):
    """Calculate the average dipole tilt angle over a whole calendar day for a datetime.

    Args:
    ----
        time (datetime): Datetime in UT for which to perform calculation.

    Returns:
    -------
        (float): Diurnally-averaged dipole tilt angle in radians.

    """
    times = [
        dt.datetime(time.year, time.month, time.day, 0, 0, 0) + dt.timedelta(hours=i)
        for i in range(0, 25)
    ]
    M_x, M_y, M_z = calc_dipole_axis(np.array(times), coords="GSM")
    mu = np.arctan(M_x / np.sqrt(M_y**2 + M_z**2))

    return np.mean(mu)


def subsolar_angles(time, coords="MAG"):
    """Calculate the latitude and longitude of the subsolar point.

    Calculates the latitude and longitude of the subsolar line at a given datetime in
    either geographic or geomagnetic coordinates.

    Args:
    ----
        time (datetime): Datetime in UT for which to perform calculation.
        coords (str): Coordinate system (GEO or MAG) in which to calculate angles.

    Returns:
    -------
        (float, float): Latitude and longitude of subsolar line in radians.

    """
    ss_x, ss_y, ss_z = GEI_to_GSE(1, 0, 0, time, inv=True)
    ss_x, ss_y, ss_z = GEO_to_GEI(ss_x, ss_y, ss_z, time, inv=True)
    ss_th, ss_az = cart_to_polar(ss_x, ss_y, ss_z)
    if coords == "MAG":
        ss_th, ss_az = GEO_to_MAG(ss_th, ss_az, time)
        return ss_th, ss_az
    elif coords == "GEO":
        return ss_th, ss_az
    else:
        print("Please specify either MAG or GEO coordinates.")


def polar_to_cart(lat, lon):
    """Transform spherical polar into cartesian coordinates on a unit sphere.

    Args:
    ----
        lat (np.array): Latitude in radians.
        lon (np.array): Longitude in radians.

    Returns:
    -------
        (np.array, np.array. np.array): Resulting cartesian coordinates (X, Y, Z).

    """
    X = np.cos(lat) * np.cos(lon)
    Y = np.cos(lat) * np.sin(lon)
    Z = np.sin(lat)

    return X, Y, Z


def cart_to_polar(X, Y, Z):
    """Transform cartesian coordinates on unit sphere into spherical polar coordinates.

    Args:
    ----
        X (np.array): Cartesian X-coordinate.
        Y (np.array): Cartesian Y-coordinate.
        Z (np.array): Cartesian Z-coordinate.

    Returns:
    -------
        (np.array, np.array): Resulting spherical polar coordinates
        (latitude, longitude)

    """
    lat = np.arctan2(Z, np.sqrt(X**2 + Y**2))
    lon = np.arctan2(Y, X)

    lon = np.where(lon < 0, 2 * np.pi + lon, lon)

    return lat, lon


def cart_to_sph(x, y, z):
    """Transform co-ordinates of a point from Cartesian to spherical polar co-ordinates.

    Args:
    ----
        x (integer/float/np.array): cartesian x-coordinate of the point.
        y (integer/float/np.array): cartesian y-coordinate of the point.
        z (integer/float/np.array): cartesian z-coordinate of the point.

    Returns:
    -------
        (integer/float/np.array,integer/float/np.array,integer/float/np.array):
        Resulting spherical polar coordinates of the point
        (radius, colatitude, azimuth).

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(z / r)
    az = np.arctan2(y, x)

    return r, th, az


def sph_to_cart(rad, pol, azi):
    """Transform spherical polar to Cartesian co-ordinates.

    Args:
    ----
        rad (integer/float/np.array): radial component of point in spherical polar
        co-ordinates.
        pol (integer/float/np.array): polar angle of point in spherical polar
        co-ordinatess, as defined from z-axis (0 to np.pi).
        azi (integer/float/np.array): azimuthal angle of point in spherical polar
        co-ordinates, as defined from x-axis (0 to 2np.pi).

    Returns:
    -------
        (integer/float/np.array,integer/float/np.array,integer/float/np.array):
        Resulting Cartesian co-ordinates of the point (x, y, z).

    """
    x = rad * np.sin(pol) * np.cos(azi)
    y = rad * np.sin(pol) * np.sin(azi)
    z = rad * np.cos(pol)

    return x, y, z


def cart_to_sph_vec(v_x, v_y, v_z, pol, azi):
    """Transform a vector from Cartesian into spherical polar co-ordinates.

    Args:
    ----
        v_x (integer/float/np.array): x-component of vector in Cartesian co-ordinates.
        v_y (integer/float/np.array): y-component of vector in Cartesian co-ordinates.
        v_z (integer/float/np.array): z-component of vector in Cartesian co-ordinates.
        pol (integer/float/np.array): polar angle of reference point in spherical polar
        co-ordinates, as defined from z-axis (0 to np.pi).
        azi (integer/float/np.array): azimuthal angle of reference point in spherical
        polar co-ordinates, as defined from x-axis (0 to 2np.pi).

    Returns:
    -------
        v_rad (integer/float/np.array): radial component of vector in spherical polar
        co-ordinates, would be upward given geographic co-ordinates.
        v_pol (integer/float/np.array): polar component of vector in spherical polar
        co-ordinates, would be in the southward direction given geographic co-ordinates.
        v_azi (integer/float/np.array): azimuthal component of vector in spherical polar
        co-ordinates, would be in the eastward direction given geographic co-ordinates.

    """
    v_rad = (
        np.sin(pol) * np.cos(azi) * v_x
        + np.sin(pol) * np.sin(azi) * v_y
        + np.cos(pol) * v_z
    )
    v_pol = (
        np.cos(pol) * np.cos(azi) * v_x
        + np.cos(pol) * np.sin(azi) * v_y
        - np.sin(pol) * v_z
    )
    v_azi = -np.sin(azi) * v_x + np.cos(azi) * v_y

    return v_rad, v_pol, v_azi


def sph_to_cart_vec(v_r, v_pol, v_azi, pol, azi):
    """Convert spherical coordinates to cartesian coordinates.

    Args:
    ----
        v_r (float): radial component of the vector
        v_pol (float): polar component of the vector
        v_azi (float): azimuthal component of the vector
        pol (float): polar angle in radians
        azi (float): azimuthal angle in radians

    Returns:
    -------
        tuple: x, y, and z components of the vector in cartesian coordinates

    """
    v_x = (
        (v_r * np.sin(pol) * np.cos(azi))
        + (v_pol * np.cos(pol) * np.cos(azi))
        - (v_azi * np.sin(azi))
    )
    v_y = (
        (v_r * np.sin(pol) * np.sin(azi))
        + (v_pol * np.cos(pol) * np.sin(azi))
        + (v_azi * np.cos(azi))
    )
    v_z = (v_r * np.cos(pol)) - (v_pol * np.sin(pol))
    return v_x, v_y, v_z


def cart2nez(x, y, z):
    """Convert Cartesian coordinates to negative `x`.

    Args:
    ----
        x (float): The x-coordinate in the Cartesian system.
        y (float): The y-coordinate in the Cartesian system.
        z (float): The z-coordinate in the Cartesian system.

    Returns:
    -------
        Tuple[float, float, float]: The converted coordinates.

    """
    return -x, y, z


def rot_x(x1, y1, z1, theta):
    """Rotates an arbitrary cartesian vector about the X-axis by an angle theta.

    Args:
    ----
        x1 (np.array): Cartesian vector X-component.
        y1 (np.array): Cartesian vector Y-component.
        z1 (np.array): Cartesian vector Z-component.
        theta (np.array): Angle in radians through which to rotate.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting rotated cartesian vector components
        (v_x, v_y, v_z).

    """
    x2 = 1 * x1
    y2 = np.cos(theta) * y1 + np.sin(theta) * z1
    z2 = -np.sin(theta) * y1 + np.cos(theta) * z1

    return x2, y2, z2


def rot_y(x1, y1, z1, theta):
    """Rotates an arbitrary cartesian vector about the Y-axis by an angle theta.

    Args:
    ----
        x1 (np.array): Cartesian vector X-component.
        y1 (np.array): Cartesian vector Y-component.
        z1 (np.array): Cartesian vector Z-component.
        theta (np.array): Angle in radians through which to rotate.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting rotated cartesian vector components
        (v_x, v_y, v_z).

    """
    x2 = np.cos(theta) * x1 + np.sin(theta) * z1
    y2 = 1 * y1
    z2 = -np.sin(theta) * x1 + np.cos(theta) * z1

    return x2, y2, z2


def rot_z(x1, y1, z1, theta):
    """Rotates an arbitrary cartesian vector about the X-axis by an angle theta.

    Args:
    ----
        x1 (np.array): Cartesian vector X-component.
        y1 (np.array): Cartesian vector Y-component.
        z1 (np.array): Cartesian vector Z-component.
        theta (np.array): Angle in radians through which to rotate.

    Returns:
    -------
        (np.array, np.array, np.array): Resulting rotated cartesian vector components
        (v_x, v_y, v_z).

    """
    x2 = np.cos(theta) * x1 + np.sin(theta) * y1
    y2 = -np.sin(theta) * x1 + np.cos(theta) * y1
    z2 = 1 * z1

    return x2, y2, z2


def plot_gorgon_axes(
    time, v_SW=[-1, 0, 0], B_IMF=[0, 0, -1], coords="GSE", filename=None, disp=True
):
    """Plot the primary axes in the simulation domain.

    Args:
    ----
        time (datetime): Date and time at which to plot axes.
        v_SW (list, optional): Solar wind inflow unit vector. Defaults to [-1,0,0].
        B_IMF (list, optional): IMF unit vector. Defaults to [0,0,-1].
        coords (str, optional): Coordinate system to use out of 'GSE', 'GSM' or 'SM'.
        Defaults to 'GSE'.
        filename (str, optional): Name of the file if you wish to save the output,
        if None then no file will be saved. Defaults to None.
        disp (bool, optional): Choose whether to display the plot in case you just wish
        to save the output. Defaults to True.

    """
    import matplotlib.pyplot as plt

    # Dipole axis vector
    M_x, M_y, M_z = calc_dipole_axis(time, coords=coords)

    # Rotation axis vector
    R_x, R_y, R_z = GEO_to_GEI(0, 0, 1, time)
    R_x, R_y, R_z = GEI_to_GSE(R_x, R_y, R_z, time)
    if coords in ["GSM", "SM"]:
        R_x, R_y, R_z = GSE_to_GSM(R_x, R_y, R_z, time)
        if coords == "SM":
            R_x, R_y, R_z = GSM_to_SM(R_x, R_y, R_z, time)

    # Solar wind vectors
    v_x, v_y, v_z = v_SW[0], v_SW[1], v_SW[2]
    B_x, B_y, B_z = B_IMF[0], B_IMF[1], B_IMF[2]
    if coords in ["GSM", "SM"]:
        v_x, v_y, v_z = GSE_to_GSM(v_x, v_y, v_z, time)
        B_x, B_y, B_z = GSE_to_GSM(B_x, B_y, B_z, time)
        if coords == "SM":
            v_x, v_y, v_z = GSM_to_SM(v_x, v_y, v_z, time)
            B_x, B_y, B_z = GSM_to_SM(B_x, B_y, B_z, time)

    R_scale = 10  # Length of axis vectors in R_E when drawn
    elev, azim = 30, -60  # Viewing projection

    # Rank in order of y-extent (for z-ordering in plot)
    if coords == "SM":
        ax_obs = np.array([-M_y + 0.001, -R_y, 0, -1, 0])
    else:
        ax_obs = np.array([-M_y, -R_y, 0, -1, 0])
    ax_obs_temp = ax_obs.argsort()
    obs_ranks = np.empty_like(ax_obs_temp)
    obs_ranks[ax_obs_temp] = np.arange(len(ax_obs))

    plt.style.use("default")
    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca(projection="3d")

    # Draw dipole axis
    M = Arrow3D(
        [0, M_x * R_scale],
        [0, M_y * R_scale],
        [0, M_z * R_scale],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="darkblue",
        label="Dipole",
        zorder=obs_ranks[0],
    )
    ax.add_artist(M)

    # Draw rotation axis
    R = Arrow3D(
        [0, R_x * R_scale],
        [0, R_y * R_scale],
        [0, R_z * R_scale],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="firebrick",
        label="Rotation",
        zorder=obs_ranks[1],
    )
    ax.add_artist(R)

    # Draw GSE axes
    X = Arrow3D(
        [0, R_scale],
        [0, 0],
        [0, 0],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="grey",
        zorder=obs_ranks[2],
    )
    Y = Arrow3D(
        [0, 0],
        [0, R_scale],
        [0, 0],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="grey",
        zorder=obs_ranks[3],
    )
    Z = Arrow3D(
        [0, 0],
        [0, 0],
        [0, R_scale],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="grey",
        zorder=obs_ranks[4],
    )
    ax.add_artist(X)
    ax.add_artist(Y)
    ax.add_artist(Z)

    # Draw solar wind inflow arrows
    v1 = Arrow3D(
        [30, 30 + v_x * R_scale / 2],
        [0, v_y * R_scale / 2],
        [7.5, 7.5 + v_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="orange",
    )
    v2 = Arrow3D(
        [30, 30 + v_x * R_scale / 2],
        [0, v_y * R_scale / 2],
        [0, v_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="orange",
    )
    v3 = Arrow3D(
        [30, 30 + v_x * R_scale / 2],
        [0, v_y * R_scale / 2],
        [-7.5, -7.5 + v_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="orange",
    )
    ax.add_artist(v1)
    ax.add_artist(v2)
    ax.add_artist(v3)

    # Draw IMF orientation
    B1 = Arrow3D(
        [20, 20 + B_x * R_scale / 2],
        [0, B_y * R_scale / 2],
        [7.5, 7.5 + B_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="darkgreen",
    )
    B2 = Arrow3D(
        [20, 20 + B_x * R_scale / 2],
        [0, B_y * R_scale / 2],
        [0, B_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="darkgreen",
    )
    B3 = Arrow3D(
        [20, 20 + B_x * R_scale / 2],
        [0, B_y * R_scale / 2],
        [-7.5, -7.5 + B_z * R_scale / 2],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="darkgreen",
    )
    ax.add_artist(B1)
    ax.add_artist(B2)
    ax.add_artist(B3)

    # Plot Earth
    ax.scatter([0], [0], [0], color="turquoise", s=200)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-10, 30)
    ax.set_xlabel(r"$X_{" + coords + "}$ / $R_E$", fontsize=16, labelpad=9)
    ax.set_ylim(-20, 20)
    ax.set_ylabel(r"$Y_{" + coords + "}$ / $R_E$", fontsize=16, labelpad=9)
    ax.set_zlim(-20, 20)
    ax.set_zlabel(r"$Z_{" + coords + "}$ / $R_E$", fontsize=16, labelpad=9)

    plt.plot([0], [0], color="darkblue", label="Dipole Axis")
    plt.plot([0], [0], color="firebrick", label="Rotation Axis")
    plt.plot([0], [0], color="grey", label=coords + " Axes")
    plt.plot([0], [0], color="orange", label="SW Inflow")
    plt.plot([0], [0], color="darkgreen", label="IMF")
    plt.legend(loc=2, frameon=False, fontsize=12)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S"), pad=20)
    ax.tick_params(axis="both", labelsize=12.5)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=50)

    if disp:
        plt.show()
    else:
        plt.close()


class Arrow3D(FancyArrowPatch):
    """Class for drawing 3-D arrows in matplotlib plots."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Generate a 3-D arrow object for the plot_gorgon_axes function.

        Args:
        ----
            xs (float): Cartesian vector X-component.
            ys (float): Cartesian vector Y-component.
            zs (float): Cartesian vector Z-component.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the 3-D arrow object.

        Args:
        ----
            renderer (matplotlib renderer): Renderer for the plot.

        """
        from mpl_toolkits.mplot3d import proj3d

        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
