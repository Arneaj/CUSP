"""Module for calculating derived parameters.

Created on Wed Mar 22 20:29:29 2017

@author: Lars
"""
import numpy as np
import scipy.constants as const

mp = const.proton_mass
qe = const.elementary_charge
mu_0 = const.mu_0

# Pressures


def Thermal_Pressure(rho, Ti, Te=None):
    """Calculate the thermal pressure of a plasma given its density and temperature.

    Args:
    ----
        rho (float): Plasma density in kg/m^3.
        Ti (float): Ion temperature in eV.
        Te (float, optional): Electron temperature in eV. If not provided, assumes
        Te = Ti/2.

    Returns:
    -------
        float: Thermal pressure in Pa.

    """
    if Te is None:
        T = 2 * Ti
    else:
        T = Te + Ti
    return rho * T * qe / mp


def Dynamic_Pressure(rho, v):
    """Todo: Docstring for Dynamic_Pressure."""
    return rho * v**2


def Magnetic_Pressure(B):
    """Todo: Docstring for Magnetic_Pressure."""
    return B**2 / (2 * mu_0)


# Wave Speeds


def Alfven_Speed(rho, B):
    """Todo: Docstring for Alfven_Speed."""
    return B / np.sqrt(mu_0 * rho)


def Sound_Speed(rho, Ti, Te=None, gamma=5.0 / 3):
    """Todo: Docstring for Sound_Speed."""
    return np.sqrt(gamma * Thermal_Pressure(rho, Ti, Te) / rho)


def Fast_Magnetosonic_Speed(rho, Te, Ti, B, th=0.0, gamma=5.0 / 3):
    """Todo: Docstring for Fast_Magnetosonic_Speed."""
    cs2 = Sound_Speed(rho, Ti, Te, gamma) ** 2
    vA2 = Alfven_Speed(rho, B) ** 2

    sqr_sum = vA2 + cs2

    vMS2 = 0.5 * (sqr_sum + np.sqrt(sqr_sum**2 - 4 * vA2 * cs2 * np.cos(th) ** 2))

    return np.sqrt(vMS2)


def Slow_Magnetosonic_Speed(rho, Te, Ti, B, th=0.0, gamma=5.0 / 3):
    """Todo: Docstring for Slow_Magnetosonic_Speed."""
    cs2 = Sound_Speed(rho, Ti, Te, gamma) ** 2
    vA2 = Alfven_Speed(rho, B) ** 2

    sqr_sum = vA2 + cs2

    vMS2 = 0.5 * (sqr_sum - np.sqrt(sqr_sum**2 - 4 * vA2 * cs2 * np.cos(th) ** 2))

    return np.sqrt(vMS2)


# Mach Numbers


def Alfven_Mach(rho, v, B):
    """Todo: Docstring for Alfven_Mach."""
    return v / Alfven_Speed(rho, B)


def Sound_Mach(rho, v, Ti, Te):
    """Todo: Docstring for Sound_Mach."""
    return v / Sound_Speed(rho, Ti, Te)
