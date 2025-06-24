"""Module providing functions for CIM calculations."""

import numpy as np
from scipy import signal as sig


def CIM_calc(
    length,
    height,
    impedence,
    current,
    phase,
    frequency,
    x_coord,
    y_coord,
    z_coord,
    rho1,
    ph1,
    rho2,
    ph2,
    ignore_FAC=False,
):
    """Apply the complex image method to obtain the electric and magnetic fields.

    Applies the complex image method to obtain the electric and magnetic fields due to a
    single U-shaped current filament carrying a current I along a length L,
    with angular frequency w, at a height h above the ground.

    The U-shaped current filament comprises a horizontal (electrojet) segment
    terminated by two vertical (field-aligned current) segments. The electric and
    magnetic fields due to this current filament are evaluated at a single point, but
    within 3 different coordinate systems: (x, y, z) in cartesian coordinates measured
    from the centre of the horizontal segment; (rho1, ph1) in cylindrical coordinates
    measured from the base of one of the vertical segments; and (rho2, ph1) at the base
    of the other. This is described in detail with derivations in Pirjola et al. (1998).

    Note the analytical solutions used in the functions below have been tested against
    the examples in the above paper, and show extremely close agreement.

    Args:
    ----
        length (float): Length of horizontal segment of U-shaped current filament.
        height (float): Height in metres of current filament above the ground.
        impedence (float): Complex impedence of the solid Earth in region beneath
        current filament.
        current (float or np.array): Current in Amperes carried by U-shaped filament.
        phase (float or np.array): Phase associated with given harmonic.
        frequency (float or np.array): Angular frequency of given harmonic.
        x_coord (float): Cartesian X-coordinate at which to evaluate fields, measured
        from centre of horizontal segment.
        y_coord (float): Cartesian Y-coordinate at which to evaluate fields, measured
        from centre of horizontal segment.
        z_coord (float): Cartesian Z-coordinate at which to evaluate fields, measured
        from centre of horizontal segment.
        rho1 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        ph1 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        rho2 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ph2 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ignore_FAC (bool, optional): Optionally ignore the field-aligned current
        contribution to the ground field.

    Returns:
    -------
        (np.array, np.array): Electric and magnetic field at reference point.
    """
    # Skin-depth
    p = impedence / (1j * frequency * 1.2566e-6)

    # Fields
    A = magpotcalc(
        length,
        height,
        current,
        p,
        x_coord,
        y_coord,
        z_coord,
        rho1,
        ph1,
        rho2,
        ph2,
        ignore_FAC,
    )  # Total magnetic vector potential
    E = -1j * frequency * A  # Total electric field
    B = Bfieldcalc(
        length,
        height,
        current,
        p,
        x_coord,
        y_coord,
        z_coord,
        rho1,
        ph1,
        rho2,
        ph2,
        ignore_FAC,
    )  # Total magnetic field

    # Phase-lagged fields
    E = E * np.exp(1j * phase)
    B = B * np.exp(1j * phase)

    return E, B


def magpotcalc(L, h, current, p, x, y, z, rho1, ph1, rho2, ph2, ignore_FAC=False):
    """Calculate and sums the components of the magnetic vector potential.

    The CIM calculations are evaluated at the reference point for each segment of the
    U-shaped current filament.

    Args:
    ----
        L (float): Length of horizontal segment of U-shaped current filament.
        h (float): Height in metres of current filament above the ground.
        current (float or np.array): Current in Amperes carried by U-shaped filament.
        p (float or np.array): Skin-depth of solid Earth.
        x (float): Cartesian X-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        y (float): Cartesian Y-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        z (float): Cartesian Z-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        rho1 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        ph1 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        rho2 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ph2 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ignore_FAC (bool, optional): Optionally ignore the field-aligned current
        contribution to the ground field.

    Returns:
    -------
        np.array: Magnetic vector potential at reference point.
    """
    Ah_el, Ah_im_el = Electrojet_magpot_calc(current, h, p, L, x, y, z)  # Electrojet
    if not ignore_FAC:
        Ah_FAC1, Ah_im_FAC1, Av_FAC1 = FAC_magpot_calc(
            current, h, p, z, rho1, ph1
        )  # FAC filament 1
        Ah_FAC2, Ah_im_FAC2, Av_FAC2 = FAC_magpot_calc(
            -current, h, p, z, rho2, ph2
        )  # FAC filament 2
    else:
        Ah_FAC1, Ah_im_FAC1, Av_FAC1 = 0 * Ah_el, 0 * Ah_el, 0 * Ah_el  # FAC filament 1
        Ah_FAC2, Ah_im_FAC2, Av_FAC2 = 0 * Ah_el, 0 * Ah_el, 0 * Ah_el  # FAC filament 2

    # Converting cylindrical components into cartesian (FAC + image FAC only)
    Acyl_FAC1 = [Ah_FAC1 + Ah_im_FAC1, 0, Av_FAC1]
    Acyl_FAC2 = [Ah_FAC2 + Ah_im_FAC2, 0, Av_FAC2]
    Acart_FAC1 = np.array(
        cyl2cart(Acyl_FAC1[0], Acyl_FAC1[1], Acyl_FAC1[2], rho1, ph1, z)
    )
    Acart_FAC2 = np.array(
        cyl2cart(Acyl_FAC2[0], Acyl_FAC2[1], Acyl_FAC2[2], rho2, ph2, z)
    )

    A = Acart_FAC1 + Acart_FAC2 + [0 * Ah_el, Ah_el + Ah_im_el, 0 * Ah_el]

    return A


def FAC_magpot_calc(current, h, p, z, rho, ph):
    """Calculate the components of the magnetic vector potential.

    Calculate the components of the magnetic vector potential due to one the vertical
    current segments and its image current, evaluated at reference point.

    Args:
    ----
        current (np.array): Current in Amperes carried by U-shaped filament.
        h (float): Height in metres of current filament above the ground.
        p (np.array): Skin-depth of solid Earth.
        z (float): Cylindrical Z-coordinate at which to evaluate field, measured from
        base of vertical segment in question.
        rho (float): Cylindrical radial-coordinate at which to evaluate field, measured
        from base of vertical segment in question.
        ph (float): Cylindrical azimuthal-coordinate at which to evaluate field,
        measured from base of vertical segment in question.

    Returns:
    -------
        (np.array, np.array, np.array): Horizontal and vertical components of the
        magnetic vector potential.
    """
    # Magnetic vector potential due to FAC:
    # Horizontal component (along surface, points in dir. of rho)
    Ah_FAC = (
        -1.2566e-6
        * current
        / (4 * np.pi)
        * rho
        / (np.sqrt(rho**2 + (z + h) ** 2) + (z + h))
    )
    # Vertical component (perpendicular to surface, points in dir. of z)
    Av_FAC = (
        -1.2566e-6
        * current
        / (4 * np.pi)
        * np.log(np.sqrt(rho**2 + (z + h) ** 2) + (z + h))
    )

    # Magnetic vector potential due to image FAC:
    # Horizontal component (along surface, points in dir. of rho))
    Ah_im_FAC = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * rho
        / (np.sqrt(rho**2 + (z + h + 2 * p) ** 2) + (z + h + 2 * p))
    )

    return Ah_FAC, Ah_im_FAC, Av_FAC


def Electrojet_magpot_calc(current, h, p, L, x, y, z):
    """Calculate the components of the magnetic vector potential.

    Calculate the components of the magnetic vector potential due the horizontal current
    segment and its image current, evaluated at reference point.

    Args:
    ----
        current (np.array): Current in Amperes carried by U-shaped filament.
        h (float): Height in metres of current filament above the ground.
        p (np.array): Skin-depth of solid Earth.
        L (float): Length of horizontal segment of U-shaped current filament.
        x (float): Cartesian X-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        y (float): Cartesian Y-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        z (float): Cartesian Z-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.

    Returns:
    -------
        (np.array, np.array): Horizontal components of the magnetic vector potential.
    """
    # Magnetic vector potential due to electrojet:
    # Horizontal component (along surface, points in dir. of y)
    Ah_el = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * (
            np.arcsinh((L / 2 - y) / np.sqrt(x**2 + (z + h) ** 2))
            - np.arcsinh((-L / 2 - y) / np.sqrt(x**2 + (z + h) ** 2))
        )
    )

    # Component due to image electrojet:
    # Horizontal component (along surface, points in dir. of y)
    Ah_im_el = (
        -1.2566e-6
        * current
        / (4 * np.pi)
        * (
            np.arcsinh((L / 2 - y) / np.sqrt(x**2 + (z + h + 2 * p) ** 2))
            - np.arcsinh((-L / 2 - y) / np.sqrt(x**2 + (z + h + 2 * p) ** 2))
        )
    )

    return (Ah_el, Ah_im_el)


def Bfieldcalc(L, h, current, p, x, y, z, rho1, ph1, rho2, ph2, ignore_FAC=False):
    """Calculate and sums the components of the magnetic field.

    The CIM calculations are evaluated at the reference point for each segment of the
    U-shaped current filament.

    Args:
    ----
        L (float): Length of horizontal segment of U-shaped current filament.
        h (float): Height in metres of current filament above the ground.
        current (float or np.array): Current in Amperes carried by U-shaped filament.
        p (float or np.array): Skin-depth of solid Earth.
        x (float): Cartesian X-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        y (float): Cartesian Y-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        z (float): Cartesian Z-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        rho1 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        ph1 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of first vertical segment.
        rho2 (float): Cylindrical radial-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ph2 (float): Cylindrical azimuthal-coordinate at which to evaluate fields,
        measured from base of second vertical segment.
        ignore_FAC (bool, optional): Optionally ignore the field-aligned current
        contribution to the ground field.

    Returns:
    -------
        np.array: Magnetic field at reference point.
    """
    Bx_el, By_el, Bz_el = Electrojet_Bfield_calc(
        current, h, p, L, x, y, z
    )  # Electrojet
    if not ignore_FAC:
        Br_FAC1, Bph_FAC1, Bz_FAC1 = FAC_Bfield_calc(
            current, h, p, z, rho1, ph1
        )  # FAC filament 1
        Br_FAC2, Bph_FAC2, Bz_FAC2 = FAC_Bfield_calc(
            -current, h, p, z, rho2, ph2
        )  # FAC filament 2
    else:
        Br_FAC1, Bph_FAC1, Bz_FAC1 = 0 * Bx_el, 0 * Bx_el, 0 * Bx_el  # FAC filament 1
        Br_FAC2, Bph_FAC2, Bz_FAC2 = 0 * Bx_el, 0 * Bx_el, 0 * Bx_el  # FAC filament 1

    # Converting cylindrical components into cartesian (FAC + image FAC only)
    Bcart_FAC1 = np.array(cyl2cart(Br_FAC1, Bph_FAC1, Bz_FAC1, rho1, ph1, z))
    Bcart_FAC2 = np.array(cyl2cart(Br_FAC2, Bph_FAC2, Bz_FAC2, rho2, ph2, z))

    B = Bcart_FAC1 + Bcart_FAC2 + [Bx_el, By_el, Bz_el]

    return B


def FAC_Bfield_calc(current, h, p, z, rho, ph):
    """Calculate the components of the magnetic field.

    Calculate the components of the magnetic field due to one the vertical current
    segments and its image current, evaluated at reference point.

    Args:
    ----
        current (np.array): Current in Amperes carried by U-shaped filament.
        h (float): Height in metres of current filament above the ground.
        p (np.array): Skin-depth of solid Earth.
        z (float): Cylindrical Z-coordinate at which to evaluate field, measured from
        base of vertical segment in question.
        rho (float): Cylindrical radial-coordinate at which to evaluate field, measured
        from base of vertical segment in question.
        ph (float): Cylindrical azimuthal-coordinate at which to evaluate field,
        measured from base of vertical segment in question.

    Returns:
    -------
        (np.array, np.array, np.array): Cylindrical radial, azimuthal and vertical
        components of the magnetic field.
    """
    # Analytic solution to curl of vector potential in cylindrical coordinates
    # Bph = (dAp/dz)*phi_hat (only take into account the horizontal equivalent
    # current contribution)
    Bph_FAC = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * rho
        / (
            (np.sqrt(rho**2 + (z + h) ** 2))
            * (np.sqrt(rho**2 + (z + h) ** 2) + (z + h))
        )
    )
    Bph_im_FAC = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * rho
        / (
            (np.sqrt(rho**2 + (z + h + 2 * p) ** 2))
            * (np.sqrt(rho**2 + (z + h + 2 * p) ** 2) + (z + h + 2 * p))
        )
    )

    Bph = Bph_FAC + Bph_im_FAC
    Br = 0 * Bph
    Bz = 0 * Bph

    return Br, Bph, Bz


def Electrojet_Bfield_calc(current, h, p, L, x, y, z):
    """Calculate the components of the magnetic field.

    Calculate the components of the magnetic field due to the horizontal current segment
    and its image current, evaluated at reference point.

    Args:
    ----
        current (np.array): Current in Amperes carried by U-shaped filament.
        h (float): Height in metres of current filament above the ground.
        p (np.array): Skin-depth of solid Earth.
        L (float): Length of horizontal segment of U-shaped current filament.
        x (float): Cartesian X-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        y (float): Cartesian Y-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.
        z (float): Cartesian Z-coordinate at which to evaluate fields, measured from
        centre of horizontal segment.

    Returns:
    -------
        (np.array, np.array, np.array): Cartesian X, Y and Z components
        of the magnetic field.
    """
    Bx_el = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * (
            (L / 2 - y)
            * (z + h)
            / (
                (x**2 + (z + h) ** 2) ** 1.5
                * np.sqrt((L / 2 - y) ** 2 / ((z + h) ** 2 + x**2) + 1)
            )
            - (-L / 2 - y)
            * (z + h)
            / (
                (x**2 + (z + h) ** 2) ** 1.5
                * np.sqrt((-L / 2 - y) ** 2 / ((z + h) ** 2 + x**2) + 1)
            )
        )
    )
    Bz_el = (
        -1.2566e-6
        * current
        / (4 * np.pi)
        * (
            (L / 2 - y)
            * x
            / (
                (x**2 + (z + h) ** 2) ** 1.5
                * np.sqrt((L / 2 - y) ** 2 / ((z + h) ** 2 + x**2) + 1)
            )
            - (-L / 2 - y)
            * x
            / (
                (x**2 + (z + h) ** 2) ** 1.5
                * np.sqrt((-L / 2 - y) ** 2 / ((z + h) ** 2 + x**2) + 1)
            )
        )
    )

    Bx_im_el = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * (
            (L / 2 - y)
            * (z + h + 2 * p)
            / (
                (x**2 + (z + h + 2 * p) ** 2) ** 1.5
                * np.sqrt((L / 2 - y) ** 2 / ((z + h + 2 * p) ** 2 + x**2) + 1)
            )
            - (-L / 2 - y)
            * (z + h + 2 * p)
            / (
                (x**2 + (z + h + 2 * p) ** 2) ** 1.5
                * np.sqrt((-L / 2 - y) ** 2 / ((z + h + 2 * p) ** 2 + x**2) + 1)
            )
        )
    )
    Bz_im_el = (
        1.2566e-6
        * current
        / (4 * np.pi)
        * (
            (L / 2 - y)
            * x
            / (
                (x**2 + (z + h + 2 * p) ** 2) ** 1.5
                * np.sqrt((L / 2 - y) ** 2 / ((z + h + 2 * p) ** 2 + x**2) + 1)
            )
            - (-L / 2 - y)
            * x
            / (
                (x**2 + (z + h + 2 * p) ** 2) ** 1.5
                * np.sqrt((-L / 2 - y) ** 2 / ((z + h + 2 * p) ** 2 + x**2) + 1)
            )
        )
    )

    Bx = Bx_el + Bx_im_el
    Bz = Bz_el + Bz_im_el
    By = 0 * Bx

    return Bx, By, Bz


def cyl2cart(Ar, Aph, Az, r, ph, z):
    """Convert a cylindrical vector into a cartesian vector.

    Convert a cylindrical vector at given cylindrical coordinates
    into a cartesian vector.

    Args:
    ----
        Ar (float): Radial component of cylindrical vector.
        Aph (float): Azimuthal component of cylindrical vector.
        Az (float): Vertical component of cylindrical vector.
        r (float): Radial coordinate.
        ph (float): Azimuthal coordinate.
        z (float): Vertical coordinate.

    Returns:
    -------
        (np.array, np.array, np.array): Cartesian X, Y and Z vector components.
    """
    Ax = Ar * np.cos(ph) - Aph * np.sin(ph)
    Ay = Ar * np.sin(ph) + Aph * np.cos(ph)

    return Ax, Ay, Az


def get_spectrum(timerange, signal, t_window=30 * 60, disp=False):
    """Perform a short-time Fourier transform on a time-varying signal.

    Performs a short-time Fourier transfort (STFT) on a time-varying signal
    and returns the resulting spectral information.

    Args:
    ----
        timerange (np.array): Timestaps in seconds of signal.
        signal (np.array): Value of signal over time.
        t_window (int, optional): Period of sampling window. Defaults to 30*60.
        disp (bool, optional): Choose whether to display the power spectrum for
        debugging purposes. Defaults to False.

    Returns:
    -------
        t (np.array): Timestamps of resulting spectral space.
        freqs (np.array): Frequencies of resulting spectral space.
        amps (np.array): Amplitude of each bin in spectral space.
        phases (np.array): Phase of each bin in spectral space.
    """
    # Fixing timeseries to 1-second cadence
    times = np.arange(0, timerange[-1] - timerange[0] + 0.1, 1)
    from scipy import interpolate

    I_int = interpolate.interp1d((timerange - timerange[0]), signal)
    signal = I_int(times)

    # Perform STFT and get frequencies, amplitudes and phases
    f, t, Z = sig.stft(signal, 1, window="hann", nperseg=t_window)
    f[0] = 1e-30
    amps = np.abs(Z)
    phases = np.angle(Z)
    freqs = np.meshgrid(f, t)[0].T

    # Optionally plot spectrum for debugging purposes.
    if disp:
        import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(15, 5))

        # ax1 = plt.subplot(1, 2, 1)
        plt.plot(times / 60, signal / 1e6)
        plt.xlabel("Time / min")
        plt.ylabel(r"$I(t)$/ MA")
        plt.xlim(0, np.max(times / 60))

        # ax2 = plt.subplot(1, 2, 2)
        # p = plt.pcolormesh(t / 60, 1 / f[1:] / 60, amps[1:, :] / 1e6)
        # cbar = plt.colorbar(p)
        plt.ylabel("Period / min")
        plt.xlabel("Time / min")
        plt.savefig("SFTF.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    return t, freqs, amps, phases


def get_timeseries(E, B, t_window=30 * 60):
    """Get timeseires from electric and magnetic field power spectra.

    Performs an inverse short-time Fourier transfort (STFT) on electric
    and magnetic field power spectra and returns the resulting timeseries.

    Args:
    ----
        E (np.array): Electric field power spectrum.
        B (np.array): Magnetic field power spectrum.
        t_window (int, optional): Period of sampling window. Defaults to 30*60.

    Returns:
    -------
        t (np.array): Timestamps of resulting timeseries.
        E (np.array): Electric field timeseries.
        B (np.array): Magnetic field timeseries.
    """
    # Split into components for separate STFTs
    E_x, E_y, E_z = E[0, :, :], E[1, :, :], E[2, :, :]
    B_x, B_y, B_z = B[0, :, :], B[1, :, :], B[2, :, :]

    t, E_x = sig.istft(E_x, 1, window="hann", nperseg=t_window)
    _, E_y = sig.istft(E_y, 1, window="hann", nperseg=t_window)
    _, E_z = sig.istft(E_z, 1, window="hann", nperseg=t_window)
    _, B_x = sig.istft(B_x, 1, window="hann", nperseg=t_window)
    _, B_y = sig.istft(B_y, 1, window="hann", nperseg=t_window)
    _, B_z = sig.istft(B_z, 1, window="hann", nperseg=t_window)

    # Only the real part is needed
    E = np.array([E_x, E_y, E_z]).real
    B = np.array([B_x, B_y, B_z]).real

    return t, E, B


def coupled_CIM(
    B_x,
    B_y,
    timeindex,
    t_window=30 * 60,
    h=100 * 1e3,
    Z=0.0011 + 0.0015j,
    cond=False,
    cond_val=1,
):
    """Invert geomagnetic field data to provide CIM coupling.

    Inversion of ground geomagnetic field data to provide CIM coupling to an external
    contribution. Assumes an infinite line current directly overhead at a height h.

    Args:
    ----
    B_x (np.array): input Bx component of the ground geomagnetic field (T)
    B_y (np.array): input By component of the ground geomagnetic field (T)
    timeindex (pandas datetimeIndex): times associated with ground geomagnetic
    field data
    t_window (int, optional): Period of sampling window, defaults to 30*60
    h (float): height of equivalent line current in m, defaults to 100km
    Z (complex, optional): Complex impedence of the solid Earth, assuming Z=mu0E/B, i.e.
    surface impedance in Ohms. Defaults to 0.0011+0.0015j Ohm for a single value, but
    realistically Z is frequency dependent (even in the homogenous Earth case).
    cond (bool, optional): If True, Z is calculated from homogenous conductivity, with
    the conductivity input passed in as Z. Defaults to False
    cond_val (float, optional): If cond=True, value of conductivity to use in mS/m.
    Defaults to 0.001 mS/m

    Returns B_x and B_y of the same form as input geomagnetic field,
    with CIM contribution.

    Usage:
    bx, by = coupled_CIM(   bx.values,by.values,bx.index,t_window=30*60,
                            h=110*1e3,cond_val=10
                        )
    """
    # constants
    mu0 = 4 * np.pi * 1e-7

    # get time index in terms of seconds
    times = (timeindex.values - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(
        1, "s"
    )

    # invert geomagnetic field data
    I_x = -2.0 * np.pi * h * B_y / mu0
    I_y = 2.0 * np.pi * h * B_x / mu0

    # convert current to frequency domain
    t, f_x, I_x, phase_x = get_spectrum(times, I_x, t_window=t_window)
    t, f_y, I_y, phase_y = get_spectrum(times, I_y, t_window=t_window)

    # get skin depth
    from .elecproject import Z_from_cond

    w = 2.0 * np.pi * f_x
    if cond:
        Z = Z_from_cond(cond_val, w)
    p = Z / (1j * w * mu0)

    # get CIM geomagnetic field in frequency domain
    B_xf = I_y * mu0 / (2.0 * np.pi) * ((1 / h) + (1 / (h + 2 * p)))
    B_yf = -I_x * mu0 / (2.0 * np.pi) * ((1 / h) + (1 / (h + 2 * p)))

    # Phase-lagged fields
    B_xf = B_xf * np.exp(1j * phase_y)
    B_yf = B_yf * np.exp(1j * phase_x)

    _, B_x, B_y = get_timeseries(
        np.repeat(B_xf[np.newaxis, :, :], 3, axis=0),
        np.repeat(B_yf[np.newaxis, :, :], 3, axis=0),
        t_window,
    )

    return B_x[0, : len(times)], B_y[0, : len(times)]
