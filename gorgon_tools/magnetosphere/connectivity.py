"""Module for calculating the connectivity of a magnetosphere."""
from ._fortran import streamtracer


def calc_connectivity(x0, arr, d, xc, ns=10000, ds=None):
    """Calculate the connectivity array.

    Calculates the connectivity array for a given starting point and magnetic
    field data.

    Args:
    ----
        x0 (numpy.ndarray): Starting point for the calculation.
        arr (numpy.ndarray): Magnetic field data.
        d (numpy.ndarray): Grid spacing for the magnetic field data.
        xc (numpy.ndarray): Center of the magnetic field data.
        ns (int, optional): Number of streamlines to trace. Defaults to 10000.
        ds (float, optional): Step size for the streamlines. If None, defaults to 0.1
        times the first element of d.

    Returns:
    -------
        numpy.ndarray: The connectivity array for the given starting point and magnetic
        field data.

    """
    if ds is None:
        ds = 0.1 * d[0]

    streamtracer.ns = ns
    streamtracer.ds = ds
    streamtracer.r_IB = 1.0
    streamtracer.xc = xc
    streamtracer.inner_boundary = True

    return streamtracer.connectivity_array(x0 + xc, arr, d)
