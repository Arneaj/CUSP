"""Module for calculating the bow shock."""
import numpy as np


def BS_Farris1994(th, D_OB, M, gamma=5.0 / 3):
    """Todo: Docstring for BS_Farris1994."""
    r = ((gamma - 1) * M**2 + 2) / ((gamma + 1) * (M**2 - 1))

    if isinstance(th, np.ndarray):
        RC = radius_of_curvature(th, D_OB)
        return D_OB[1:-1] + RC * 0.8 * r
    else:
        return D_OB + 1.1 * r


def BS_GasDyn(D_OB, M, gamma=5.0 / 3):
    """Gas Dynamic Bow Shock (From Maksimovic/Spreiter)."""
    r = ((gamma - 1) * M**2 + 2) / ((gamma + 1) * (M**2 - 1))
    return D_OB * (1 + 1.1 * r)


def radius_of_curvature(th, r):
    """Todo: Docstring for radius_of_curvature."""
    from numpy import abs, sqrt

    dth = abs(th[1] - th[0])

    dr_dth = (r[2:] - r[:-2]) / (2 * dth)
    d2r_dth2 = (r[2:] - 2 * r[1:-1] + r[:-2]) / dth**2

    A = sqrt(r[1:-1] ** 2 + dr_dth**2) ** 3
    B = abs(r[1:-1] ** 2 + 2 * dr_dth**2 - r[1:-1] * d2r_dth2)

    return A / B


def BS_Jerab05(phi, th, Pd, Ma, B, gamma=5.0 / 3):
    """Jerab et. al 2005 bow shock model.

    Args:
    ----
        phi (float): Azimuth from +y toward +z
        th (float): colatitude from +x
        Pd (float): Dynamic pressure in nPa
        Ma (float): The Alfven Mach Number
        B (float): IMF strength in nT
        gamma (float, optional): The adiabatic index. Defaults to 5.0/3.

    Returns:
    -------
        float: The bow shock radius in R0

    """
    a11 = 0.45
    a22 = 1.0
    a33 = 0.8
    a12 = 0.18
    a14 = 46.6
    a24 = -2.2
    a34 = -0.6
    a44 = -618.0

    def Rav(phi, th):
        x = np.cos(th)
        y = np.sin(th) * np.sin(phi)
        z = np.sin(th) * np.cos(phi)

        a = a11 * x**2 + a22 * y**2 + a33 * z**2 + a12 * x * y
        b = a14 * x + a24 * y + a34 * z
        c = a44
        return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    D = 0.937 * (0.846 + 0.042 * B)
    C = 91.55

    R0 = Rav(0, 0)

    A = C * (1.67e-6 / Pd) ** (1 / 6)
    m = 1 + D * ((gamma - 1) * Ma**2 + 2) / ((gamma + 1) * (Ma**2 - 1))

    return Rav(phi, th) / R0 * A * m


def BS_Merka05(phi, th, n_sw, v_sw, Ma):
    """Merka et. al 2005 (MA) bow shock model (GPE coordinates).

    Args:
    ----
        phi (float): Azimuth from +y toward +z
        th (float): colatitude from +x
        n_sw (float): SW number density in cm-3
        v_sw (float): SW speed in km/s
        Ma (float): The Alfven Mach Number

    Returns:
    -------
        float: The bow shock radius in R0

    """
    from numpy import cos, sin

    # MA dependence
    b11 = [0.0063, 0.1649]
    b12 = [-0.0098, 0.0196]
    b31 = [0.8351, 0.0973]
    b32 = [0.0102, 0.01]
    b41 = [-0.0298, 0.0627]
    b42 = [0.004, 0.0072]
    b71 = [16.39, 2.94]
    b72 = [0.2087, 0.228]
    b73 = [108.3, 43.1]
    b81 = [-0.9241, 0.5913]
    b82 = [0.0721, 0.0591]
    b101 = [-444, 59.2]
    b102 = [-2.935, 4.834]
    b103 = [-1930, 618]

    a1 = b11[0] + b12[0] * Ma
    a2 = 1.0
    a3 = b31[0] + b32[0] * Ma
    a4 = b41[0] + b42[0] * Ma
    a7 = b71[0] + b72[0] * Ma + b73[0] / (Ma - 1) ** 2
    a8 = b81[0] + b82[0] * Ma
    a10 = b101[0] + b102[0] * Ma + b103[0] / (Ma - 1) ** 2

    def cos2(x):
        return cos(x) ** 2

    def sin2(x):
        return sin(x) ** 2

    A = (
        a1 * cos2(th)
        + sin2(th) * (a2 * cos2(phi) + a3 * sin2(phi))
        + a4 * sin(2 * th) * cos(phi)
    )

    B = 2 * (a7 * cos(th) + a8 * sin(th) * cos(phi))

    C = a10

    #     print("th", np.degrees(th))
    #     print("phi", np.degrees(phi))

    #     print("A", A)
    #     print("B", B)
    #     print("C", C)

    det = B**2 - 4 * A * C
    if (det < 0).any():
        print(det)

    r = np.maximum(-B + np.sqrt(det), -B - np.sqrt(det))
    r = r / (2 * A)

    # Scale back
    rs = ((n_sw * v_sw**2) / (7 * 457.5**2)) ** (1.0 / 6)

    return r * rs
