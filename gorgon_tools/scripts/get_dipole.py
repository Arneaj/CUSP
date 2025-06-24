"""Get dipole tilt for simulation relative to the time range in question."""
import datetime as dt
import sys

import numpy as np

from gorgon_tools.geomagnetic import coordinates


def get_dipole(t0_UT, gorgonops=True):
    """Get dipole tilt for simulation relative to the time range in question.

    Args:
    ----
        t0_UT (numpy.ndarray): Time in UT.
        gorgonops (bool): If True, use rounding relevant to GorgonOps. Defaults to True.

    Returns:
    -------
        float: Dipole tilt (rounded off).

    """
    t0_UT = np.array([t0_UT])
    M_x, M_y, M_z = coordinates.calc_dipole_axis(t0_UT, coords="GSM")
    _, _, _, dip_tilt, _ = coordinates.SM_to_SMD(
        M_x, M_y, M_z, t0_UT, gorgonops=gorgonops
    )
    if gorgonops:
        dip_tilt = np.round(dip_tilt)
        if dip_tilt < 0:
            dip_tilt = 360 + dip_tilt
    return dip_tilt


if __name__ == "__main__":
    # create help message for command line arguments
    if len(sys.argv) == 1:
        print(
            "Usage: python get_dipole.py <YYYY-MM-DD_HH:MM:SS> <optional: get_init_fnm>"
        )
        sys.exit()
    t0_UT = dt.datetime.strptime(
        sys.argv[1], "%Y-%m-%d_%H:%M:%S"
    )  # UT time corresponding to zero simulation time
    if len(sys.argv) > 2:
        init_fnm = str(int(get_dipole(t0_UT))).zfill(3) + "deg"
        print(init_fnm)
