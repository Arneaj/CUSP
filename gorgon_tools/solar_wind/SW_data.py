"""Module to provide utility functions for working with solar wind data."""

import datetime as dt
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.visualization import quantity_support
from scipy.constants import Boltzmann, elementary_charge, proton_mass
from sunpy.net import Fido, attrs
from sunpy.timeseries import TimeSeries

from ..geomagnetic.coordinates import (
    GEI_to_GSE,
    GEO_to_GEI,
    GSE_to_GSM,
    GSE_to_SM,
    GSM_to_SM,
    SM_to_SMD,
    calc_dipole_axis,
)

warnings.filterwarnings("ignore")

quantity_support()


def get_SW_data_raw(starttime, endtime, source):
    """Download solar wind data from the CDAWeb (https://cdaweb.gsfc.nasa.gov/).

    Args:
    ----
        starttime: datetime object
        endtime: datetime object
        source: string, CDAWeb source name

    Returns:
    -------
        sw_data: sunpy.timeseries.TimeSeries object

    """
    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset(source)
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result)
    if len(downloaded_files) == 0:
        print("No data available")
    else:
        data = TimeSeries(downloaded_files, concatenate=True)
        for file in downloaded_files:
            try:
                Path(file).unlink()
            except OSError as e:
                print(f"File '{file}' could not be removed. Ignoring. {e}")
        return data


def get_SW_data(starttime, endtime, source="OMNI"):
    """Get solar wind data from the CDAWeb (https://cdaweb.gsfc.nasa.gov/).

    Downloads solar wind data using SunPy from either ACE, DSCOVR, Wind or OMNI,
    extracts variables needed for Gorgon input at a resampled 1 min cadence, and returns
    resulting dataframe.

    The variables are in GSE coordinates and are as follows: (proton) number
    density (/cm^3), (proton) temperature (eV), velocity components and magnitude (km/s)
    and IMF components and magnitude (nT).

    Args:
    ----
        starttime (datetime): Date and time in UT for start of chosen data time window.
        endtime (datetime): Date and time in UT for end of chosen data time window.
        source (str, optional): Data source (i.e. spacecraft) to be used.
        Defaults to 'OMNI'.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data in GSE coordinates.

    """
    pd.options.mode.chained_assignment = None  # suppress irrelevant pandas warnings

    SW_data = pd.DataFrame()

    if source == "OMNI":
        # High cadence (60s) bow shock-shifted OMNI plasma and mag data
        OMNI_data = get_SW_data_raw(starttime, endtime, "OMNI_HRO_1MIN")
        if OMNI_data is None:
            print("No data available")
        else:
            SW_data = OMNI_data.data[
                [
                    "proton_density",
                    "T",
                    "Vx",
                    "Vy",
                    "Vz",
                    "flow_speed",
                    "BX_GSE",
                    "BY_GSE",
                    "BZ_GSE",
                ]
            ]
            SW_data["T"] = SW_data["T"] * Boltzmann / elementary_charge
            SW_data["B"] = np.sqrt(
                SW_data["BX_GSE"] ** 2 + SW_data["BY_GSE"] ** 2 + SW_data["BZ_GSE"] ** 2
            )
            SW_data = SW_data.rename(
                columns={
                    "proton_density": "n",
                    "T": "Ti",
                    "Vx": "vx",
                    "Vy": "vy",
                    "Vz": "vz",
                    "flow_speed": "v",
                    "BX_GSE": "Bx",
                    "BY_GSE": "By",
                    "BZ_GSE": "Bz",
                    "B": "B",
                }
            )

            SW_data = SW_data.resample("60S").mean()
            SW_data = SW_data[starttime:endtime]

    elif source == "ACE":
        # High cadence (64s) plasma data for vx, vy, vz, v, n, Ti
        plas = get_SW_data_raw(starttime, endtime, "AC_H0_SWE")
        if plas is not None:
            plas.data[abs(plas.data) > 1e20] = np.nan

        # Low cadence (12m) plasma data to cover gaps in high cadence data for
        # n - upsampled to 64s
        plas_2 = get_SW_data_raw(starttime, endtime, "AC_H6_SWI")
        if plas_2 is not None and plas is not None:
            plas_2.data[abs(plas_2.data) > 1e20] = np.nan
            plas_2.data = (
                plas_2.data.resample("64S").mean().interpolate(method="linear")
            )
            plas_2.data = plas_2.data.reindex(plas.data.index, method="nearest")
            plas.data.Np[plas.data.Np.isnull()] = plas_2.data.nH[plas.data.Np.isnull()]

        # High cadence (16s) mag data for Bx, By, Bz, B - downsampled to 64s
        mag = get_SW_data_raw(starttime, endtime, "AC_H0_MFI")
        if mag is not None and plas is not None:
            mag.data = mag.data.resample("64S").mean()
            mag.data[abs(mag.data) > 1e20] = np.nan
            mag.data = mag.data.reindex(plas.data.index, method="nearest")

            # Create new dataframe containing only Gorgon SW variables
            SW_data = pd.DataFrame(
                [
                    plas.data.Np,
                    plas.data.Tpr * Boltzmann / elementary_charge,
                    plas.data.V_GSE_0,
                    plas.data.V_GSE_1,
                    plas.data.V_GSE_2,
                    plas.data.Vp,
                    mag.data.BGSEc_0,
                    mag.data.BGSEc_1,
                    mag.data.BGSEc_2,
                    mag.data.Magnitude,
                ]
            ).transpose()
            SW_data = SW_data.rename(
                columns={
                    "Np": "n",
                    "Tpr": "Ti",
                    "V_GSE_0": "vx",
                    "V_GSE_1": "vy",
                    "V_GSE_2": "vz",
                    "Vp": "v",
                    "BGSEc_0": "Bx",
                    "BGSEc_1": "By",
                    "BGSEc_2": "Bz",
                    "Magnitude": "B",
                }
            )

            SW_data = SW_data.resample("60S").mean()
            SW_data = SW_data[starttime:endtime]

        else:
            print("No data available")

    elif source == "Wind":
        # High cadence (60s) mag data
        mag = get_SW_data_raw(starttime, endtime, "WI_H0_MFI")
        if mag is not None:
            mag.data[abs(mag.data) == 99999.9] = np.nan

        # Low cadence (92s) plasma data
        plas = get_SW_data_raw(starttime, endtime, "WI_H1_SWE")
        if plas is not None and mag is not None:
            plas.data[abs(plas.data) == 99999.9] = np.nan
            plas.data = plas.data.resample("60S").mean().interpolate(method="linear")
            plas.data = plas.data.reindex(mag.data.index, method="nearest")

            # Create new dataframe containing only Gorgon SW variables
            SW_data = pd.DataFrame(
                [
                    plas.data.Proton_Np_nonlin,
                    0.5
                    * (plas.data.Proton_W_nonlin * 1e3) ** 2
                    * proton_mass
                    / elementary_charge,
                    plas.data.Proton_VX_nonlin,
                    plas.data.Proton_VY_nonlin,
                    plas.data.Proton_VZ_nonlin,
                    plas.data.Proton_V_nonlin,
                    mag.data.BGSE_0,
                    mag.data.BGSE_1,
                    mag.data.BGSE_2,
                    mag.data.BF1,
                ]
            ).transpose()
            SW_data = SW_data.rename(
                columns={
                    "Proton_Np_nonlin": "n",
                    "Proton_W_nonlin": "Ti",
                    "Proton_VX_nonlin": "vx",
                    "Proton_VY_nonlin": "vy",
                    "Proton_VZ_nonlin": "vz",
                    "Proton_V_nonlin": "v",
                    "BGSE_0": "Bx",
                    "BGSE_1": "By",
                    "BGSE_2": "Bz",
                    "BF1": "B",
                }
            )

            SW_data = SW_data.resample("60S").mean()
            SW_data = SW_data[starttime:endtime]

        else:
            print("No data available")

    elif source == "DSCOVR":
        # High cadence (1s) mag data
        mag = get_SW_data_raw(starttime, endtime, "DSCOVR_H0_MAG")
        if mag is not None:
            mag.data[abs(mag.data) == 99999.9] = np.nan
            mag.data = mag.data.resample("60S").mean().interpolate(method="linear")

        # Low cadence (60s) plasma data
        plas = get_SW_data_raw(starttime, endtime, "DSCOVR_H1_FC")
        if plas is not None and mag is not None:
            plas.data[abs(plas.data) == 99999.9] = np.nan
            plas.data = plas.data.reindex(mag.data.index, method="nearest")

            # Create new dataframe containing only Gorgon SW variables
            SW_data = pd.DataFrame(
                [
                    plas.data.Np,
                    plas.data.THERMAL_TEMP * Boltzmann / elementary_charge,
                    plas.data.V_GSE_0,
                    plas.data.V_GSE_1,
                    plas.data.V_GSE_2,
                    mag.data.B1GSE_0,
                    mag.data.B1GSE_1,
                    mag.data.B1GSE_2,
                    mag.data.B1F1,
                ]
            ).transpose()
            SW_data = SW_data.rename(
                columns={
                    "Np": "n",
                    "THERMAL_TEMP": "Ti",
                    "V_GSE_0": "vx",
                    "V_GSE_1": "vy",
                    "V_GSE_2": "vz",
                    "B1GSE_0": "Bx",
                    "B1GSE_1": "By",
                    "B1GSE_2": "Bz",
                    "B1F1": "B",
                }
            )
            SW_data["v"] = np.sqrt(SW_data.vx**2 + SW_data.vy**2 + SW_data.vz**2)

            SW_data = SW_data.resample("60S").mean()
            SW_data = SW_data[starttime:endtime]

    else:
        print("Please choose one of OMNI, DSCOVR, ACE or Wind as the data source.")
        return

    return SW_data


def read_OMNIWeb_OMNI_data(file):
    """Generate a DataFrame equivalent to that of 'get_SW_data' by reading a .lst file.

    Generates a DataFrame equivalent to that from 'get_SW_data' by reading a .lst file
    downloaded directly from OMNIWeb (https://omniweb.gsfc.nasa.gov/form/omni_min.html).

    This file must only contain the following chosen fields, for 1-min avg data:
    'IMF Magnitude Avg', 'Bx, GSE/GSM', 'By, GSE', 'Bz, GSE', 'Flow Speed',
    'Vx Velocity, GSE', 'Vy Velocity, GSE', 'Vz Velocity, GSE', 'Proton Density',
    'Proton Temperature'.

    Args:
    ----
        file (str): Path to the .lst data file.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data in GSE coordinates.

    """
    pd.options.mode.chained_assignment = None  # suppress irrelevant pandas warnings

    dat = np.loadtxt(file)
    dates = []
    for i in range(len(dat[:, 0])):
        dates.append(
            dt.datetime(int(dat[i, 0]), 1, 1, int(dat[i, 2]), int(dat[i, 3]))
            + dt.timedelta(days=int(dat[i, 1]) - 1)
        )
    OMNI_data = pd.DataFrame(dat[:, 4:])
    OMNI_data.index = dates
    OMNI_data.columns = [
        "B",
        "Bx_GSE",
        "By_GSE",
        "Bz_GSE",
        "v",
        "vx_GSE",
        "vy_GSE",
        "vz_GSE",
        "n",
        "T",
    ]

    OMNI_data["B"][OMNI_data["B"] == 9999.99] = np.nan
    OMNI_data["Bx_GSE"][OMNI_data["Bx_GSE"] == 9999.99] = np.nan
    OMNI_data["By_GSE"][OMNI_data["By_GSE"] == 9999.99] = np.nan
    OMNI_data["Bz_GSE"][OMNI_data["Bz_GSE"] == 9999.99] = np.nan
    OMNI_data["v"][OMNI_data["v"] == 99999.9] = np.nan
    OMNI_data["vx_GSE"][OMNI_data["vx_GSE"] == 99999.9] = np.nan
    OMNI_data["vy_GSE"][OMNI_data["vy_GSE"] == 99999.9] = np.nan
    OMNI_data["vz_GSE"][OMNI_data["vz_GSE"] == 99999.9] = np.nan
    OMNI_data["n"][OMNI_data["n"] == 999.99] = np.nan
    OMNI_data["T"][OMNI_data["T"] == 9999999.0] = np.nan

    # Create new dataframe containing only Gorgon SW variables
    SW_data = pd.DataFrame(
        [
            OMNI_data["n"],
            OMNI_data["T"] * Boltzmann / elementary_charge,
            OMNI_data["vx_GSE"],
            OMNI_data["vy_GSE"],
            OMNI_data["vz_GSE"],
            OMNI_data["v"],
            OMNI_data["Bx_GSE"],
            OMNI_data["By_GSE"],
            OMNI_data["Bz_GSE"],
            OMNI_data["B"],
        ]
    ).transpose()
    SW_data = SW_data.rename(
        columns={
            "n": "n",
            "T": "Ti",
            "vx_GSE": "vx",
            "vy_GSE": "vy",
            "vz_GSE": "vz",
            "v": "v",
            "Bx_GSE": "Bx",
            "By_GSE": "By",
            "Bz_GSE": "Bz",
            "BF1": "B",
        }
    )

    return SW_data


def read_ODI_OMNI_data(file):
    """Generate a DataFrame equivalent to that of 'get_SW_data' by reading a .txt file.

    Generates a DataFrame equivalent to that from 'get_SW_data' by reading a .txt file
    from ODI OMNI.

    This file must only contain the following chosen fields, for 5-min avg data:
    'Time', 'proton_density', 'v_1', 'v_2', 'v_3', 'b_gse_1', 'b_gse_2', 'b_gse_3', 't'.

    Args:
    ----
        file (str): Path to the .txt data file.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data in GSE coordinates.

    """
    pd.options.mode.chained_assignment = None  # suppress irrelevant pandas warnings

    dat = np.genfromtxt(file, str)
    dates = []
    for i in range(len(dat[1:, 0])):
        dates.append(dt.datetime.strptime(str(dat[1 + i, 0]), "%Y-%m-%dT%H:%M:%S"))
    OMNI_data = pd.DataFrame(dat[1:, 1:].astype(float))
    OMNI_data.index = dates
    OMNI_data.columns = [
        "n",
        "vx_GSE",
        "vy_GSE",
        "vz_GSE",
        "Bx_GSE",
        "By_GSE",
        "Bz_GSE",
        "T",
    ]

    OMNI_data["v"] = np.linalg.norm(
        [OMNI_data["vx_GSE"], OMNI_data["vy_GSE"], OMNI_data["vz_GSE"]], axis=0
    )
    OMNI_data["B"] = np.linalg.norm(
        [OMNI_data["Bx_GSE"], OMNI_data["By_GSE"], OMNI_data["Bz_GSE"]], axis=0
    )

    # Create new dataframe containing only Gorgon SW variables
    SW_data = pd.DataFrame(
        [
            OMNI_data["n"],
            OMNI_data["T"] * Boltzmann / elementary_charge,
            OMNI_data["vx_GSE"],
            OMNI_data["vy_GSE"],
            OMNI_data["vz_GSE"],
            OMNI_data["v"],
            OMNI_data["Bx_GSE"],
            OMNI_data["By_GSE"],
            OMNI_data["Bz_GSE"],
            OMNI_data["B"],
        ]
    ).transpose()
    SW_data.columns = ["n", "Ti", "vx", "vy", "vz", "v", "Bx", "By", "Bz", "B"]

    return SW_data


def read_EUHFORIA_data(file):
    """Generate a DataFrame equivalent to that of 'get_SW_data' by reading a .dsv file.

    Generates a DataFrame equivalent to that from 'get_SW_data' by reading a .dsv file
    generated by the EUHFORIA heliospheric MHD model.

    The file should be provided in the HEEQ coordinate system, using the transformation:
    r -> -x_GSE, lon -> y_GSE

    Args:
    ----
        file (str): Path to the .dsv data file.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data in GSE coordinates.

    """
    pd.options.mode.chained_assignment = None  # suppress irrelevant pandas warnings

    dat = np.genfromtxt(file, str)
    dates = []
    for i in range(len(dat[1:, 0])):
        dates.append(dt.datetime.strptime(str(dat[1 + i, 0]), "%Y-%m-%dT%H:%M:%S"))
    EUHFORIA_data = pd.DataFrame(dat[1:, 4:].astype(float))
    EUHFORIA_data.index = dates
    EUHFORIA_data.columns = dat[0, 4:]

    EUHFORIA_data["v"] = np.linalg.norm(
        [
            EUHFORIA_data["vr[km/s]"],
            EUHFORIA_data["vlon[km/s]"],
            EUHFORIA_data["vclt[km/s]"],
        ],
        axis=0,
    )
    EUHFORIA_data["B"] = np.linalg.norm(
        [EUHFORIA_data["Br[nT]"], EUHFORIA_data["Blon[nT]"], EUHFORIA_data["Bclt[nT]"]],
        axis=0,
    )

    # Create new dataframe containing only Gorgon SW variables
    SW_data = pd.DataFrame(
        [
            EUHFORIA_data["n[1/cm^3]"] / 2,
            EUHFORIA_data["P[Pa]"]
            / (EUHFORIA_data["n[1/cm^3]"] * elementary_charge * 1e6),
            -EUHFORIA_data["vr[km/s]"],
            -EUHFORIA_data["vlon[km/s]"],
            -EUHFORIA_data["vclt[km/s]"],
            EUHFORIA_data["v"],
            -EUHFORIA_data["Br[nT]"],
            -EUHFORIA_data["Blon[nT]"],
            -EUHFORIA_data["Bclt[nT]"],
            EUHFORIA_data["B"],
        ]
    ).transpose()
    SW_data.columns = ["n", "Ti", "vx", "vy", "vz", "v", "Bx", "By", "Bz", "B"]

    return SW_data


def init_transition(init_data, SW_data):
    """Initialise solar wind data from a predefined state.

    Function to spin up solar wind data from initialised values. Relies on the length of
    required solar wind data dataframe and a predefined initialised state passed as an
    array.

    Input array items assumed to be simulation coordinates and in the form ['n' (/cc),
    'Ti' (eV), 'vx' (km/s), 'vy', 'vz', 'Bx' (nT), 'By', 'Bz'], for example,

    init_data=[5e6/1e6, 5., 400e3/1e3, 0., 0., 0., 0., 2e-9*1e9]

    Updates solar wind data inplace and of the same length as the input.
    """
    dat = SW_data.copy()
    parameters = ["n", "Ti", "vx", "vy", "vz", "Bx", "By", "Bz"]
    corr = np.arange(0, len(dat), 1)
    for i in range(len(parameters)):
        dat[parameters[i]] = (
            dat[parameters[i]]
            + corr
            * (
                ((dat.iloc[-1][parameters[i]] - init_data[i]) / (len(dat) - 1))
                - (
                    (dat.iloc[-1][parameters[i]] - dat.iloc[0][parameters[i]])
                    / (len(dat) - 1)
                )
            )
            - dat.iloc[0][parameters[i]]
            + init_data[i]
        )
    dat["B"] = (dat["Bx"] ** 2 + dat["By"] ** 2 + dat["Bz"] ** 2) ** 0.5
    dat["v"] = (dat["vx"] ** 2 + dat["vy"] ** 2 + dat["vz"] ** 2) ** 0.5

    return dat


def arr_time(index):
    """Convert a pandas index to an array of times in seconds."""
    return np.array([t for t in index])


def gen_gorgon_SW(
    SW_data,
    simtime=0,
    coords="GSE",
    use_SMD=True,
    fill_gaps=True,
    max_yz_inflow=(None, None),
    write=True,
    smoothing=False,
    mean_Bx=True,
    SW_append=False,
    out_dir="./",
    fnm="SW_Input.dat",
    gorgonops=False,
    init_data=None,
):
    """Generate a solar wind dataset.

    Generates a solar wind dataset in a chosen coordinate system which matches the
    requirements of the Gorgon input file, with the option to write out this file.

    Performs coordinate transformations from the default GSE frame into an arbitrary
    frame for later plotting, but output file is always in Gorgon simulation
    coordinates.

    Args:
    ----
        SW_data ([type]): Pandas DataFrame from either the 'get_SW_data' or
        'read_OMNI_data' functions.
        coords (str, optional): Coordinate system out of 'GSE', 'GSM', 'SM', 'SMD', and
        'sim' in which to return data. Defaults to 'GSE'.
        use_SMD (bool, optional): Choose whether to base the Gorgon input on
        diurnally-averaged SMD coordinates rather than SM.
        fill_gaps (bool, optional): Choose whether to interpolate between data gaps for
        later plotting. Defaults to True.
        max_yz_inflow (tuple, optional): Maximum inflow angles in degrees in the
        (X-Y, X-Z) planes (e.g. 20 for Y = [-40, 40] R_E). Defaults to (None, None).
        simtime (float, optional): Initial simulation time in seconds from which the
        solar wind data will be applied. Defaults to 0.
        write (bool, optional): Choose whether to write out the Gorgon input file using
        the data. Defaults to True. This is always in sim or SMD.
        smoothing (bool, optional): Choose whether to clean and smooth the input signal
        mean_Bx (bool, optional): Include mean of Bx over the time range.
        Defaults to True.
        SW_append (boolean, optional): Append to existing output file.
        Defaults to False.
        out_dir (string, optional): Where to output the SW data file.
        Default to current working directory, i.e., './'
        fnm (string, optional): SW data filename. Default to current 'SW_Input.dat'
        gorgonops (boolean, optional): Limit the tilt angle to increments of 5 degrees
        for GorgonOps. Defaults to False.
        init_data (list, optional): List of initial conditions to spin up from.
        Makes use of init_transition function. Further development may include defining
        the duration of the spin up. Defaults to None.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data.
        pd.DataFrame: Pandas dataframe containing processed data in simulation coords
        (only returned if 'write'=True, otherwise returns None).
        dict: Dictionary of Gorgon parameters for simulation setup.

    """
    # Rotation axis for corotation
    SW_data["Rx"], SW_data["Ry"], SW_data["Rz"] = (
        0 * SW_data.n,
        0 * SW_data.n,
        0 * SW_data.n,
    )
    SW_data.Rx, SW_data.Ry, SW_data.Rz = GEO_to_GEI(0, 0, 1, arr_time(SW_data.index))
    SW_data.Rx, SW_data.Ry, SW_data.Rz = GEI_to_GSE(
        SW_data.Rx, SW_data.Ry, SW_data.Rz, arr_time(SW_data.index)
    )

    # Get Gorgon SW inflow angle at each time; this is -1*mu, where mu is the dipole
    # tilt angle i.e. the angle between dipole axis and GSM z-axis
    SW_data["tilt"] = 0 * SW_data.n
    M_x, M_y, M_z = calc_dipole_axis(arr_time(SW_data.index), coords="GSM")
    SW_data.tilt = np.arctan(M_x / np.sqrt(M_y**2 + M_z**2))
    sim_params = {}  # Gorgon parameters to aid in setting up simulation; remains empty
    # if write = False.

    if coords == "GSM":
        SW_data.vx, SW_data.vy, SW_data.vz = GSE_to_GSM(
            SW_data.vx, SW_data.vy, SW_data.vz, arr_time(SW_data.index)
        )
        SW_data.Bx, SW_data.By, SW_data.Bz = GSE_to_GSM(
            SW_data.Bx, SW_data.By, SW_data.Bz, arr_time(SW_data.index)
        )
        SW_data.Rx, SW_data.Ry, SW_data.Rz = GSE_to_GSM(
            SW_data.Rx, SW_data.Ry, SW_data.Rz, arr_time(SW_data.index)
        )
    elif coords in ["SM", "SMD", "sim"]:
        SW_data.vx, SW_data.vy, SW_data.vz = GSE_to_SM(
            SW_data.vx, SW_data.vy, SW_data.vz, arr_time(SW_data.index)
        )
        SW_data.Bx, SW_data.By, SW_data.Bz = GSE_to_SM(
            SW_data.Bx, SW_data.By, SW_data.Bz, arr_time(SW_data.index)
        )
        SW_data.Rx, SW_data.Ry, SW_data.Rz = GSE_to_SM(
            SW_data.Rx, SW_data.Ry, SW_data.Rz, arr_time(SW_data.index)
        )
        if coords == "SMD" or (coords == "sim" and use_SMD):
            SW_data.vx, SW_data.vy, SW_data.vz, _, _ = SM_to_SMD(
                SW_data.vx,
                SW_data.vy,
                SW_data.vz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
            SW_data.Bx, SW_data.By, SW_data.Bz, _, _ = SM_to_SMD(
                SW_data.Bx,
                SW_data.By,
                SW_data.Bz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
            SW_data.Rx, SW_data.Ry, SW_data.Rz, _, _ = SM_to_SMD(
                SW_data.Rx,
                SW_data.Ry,
                SW_data.Rz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
        if coords == "sim":
            SW_data.vx, SW_data.vy, SW_data.Bx, SW_data.By, SW_data.Rx, SW_data.Ry = (
                -SW_data.vx,
                -SW_data.vy,
                -SW_data.Bx,
                -SW_data.By,
                -SW_data.Rx,
                -SW_data.Ry,
            )

    if fill_gaps:
        SW_data = SW_data.interpolate("linear")  # Pad NaNs
        SW_data = SW_data.fillna(
            method="ffill"
        )  # Fill remaining NaNs with previous value
        SW_data = SW_data.fillna(method="bfill")  # Fill remaining NaNs with next value

    if smoothing:
        pre_index = SW_data.index.union(
            pd.to_datetime(SW_data.index[0] - np.arange(1, 3) * dt.timedelta(minutes=1))
        )
        post_index = pre_index.union(
            pd.to_datetime(
                SW_data.index[-1] + np.arange(1, 3) * dt.timedelta(minutes=1)
            )
        )
        SW_data = pd.concat(
            [
                SW_data.head(1),
                SW_data.head(1),
                SW_data,
                SW_data.tail(1),
                SW_data.tail(1),
            ]
        )
        SW_data.index = post_index
        SW_data = SW_data.rolling(
            4, center=True, win_type="parzen", closed="both"
        ).mean()
        SW_data = SW_data.iloc[2:-2]

    SW_data_out = SW_data.copy()

    if write:  # Save data file in Gorgon input format
        if not fill_gaps:  # overwrites the user option for file writing
            SW_data = SW_data.interpolate(
                "linear"
            )  # Pad NaNs, i.e. the output written to file is always continuous

        # Output file must be in Gorgon coordinates even if we want to return data in
        # e.g. GSM, so transform to SM
        if coords == "GSE":
            SW_data.vx, SW_data.vy, SW_data.vz = GSE_to_SM(
                SW_data.vx, SW_data.vy, SW_data.vz, arr_time(SW_data.index)
            )
            SW_data.Bx, SW_data.By, SW_data.Bz = GSE_to_SM(
                SW_data.Bx, SW_data.By, SW_data.Bz, arr_time(SW_data.index)
            )
            SW_data.Rx, SW_data.Ry, SW_data.Rz = GSE_to_SM(
                SW_data.Rx, SW_data.Ry, SW_data.Rz, arr_time(SW_data.index)
            )
        elif coords == "GSM":
            SW_data.vx, SW_data.vy, SW_data.vz = GSM_to_SM(
                SW_data.vx, SW_data.vy, SW_data.vz, arr_time(SW_data.index)
            )
            SW_data.Bx, SW_data.By, SW_data.Bz = GSM_to_SM(
                SW_data.Bx, SW_data.By, SW_data.Bz, arr_time(SW_data.index)
            )
            SW_data.Rx, SW_data.Ry, SW_data.Rz = GSM_to_SM(
                SW_data.Rx, SW_data.Ry, SW_data.Rz, arr_time(SW_data.index)
            )

        # In Gorgon coordinates, X and Y point in the opposite sense to SM/SMD
        if coords not in ["SMD", "sim"] and use_SMD:
            SW_data.vx, SW_data.vy, SW_data.vz, _, _ = SM_to_SMD(
                SW_data.vx,
                SW_data.vy,
                SW_data.vz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
            SW_data.Bx, SW_data.By, SW_data.Bz, _, _ = SM_to_SMD(
                SW_data.Bx,
                SW_data.By,
                SW_data.Bz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
            SW_data.Rx, SW_data.Ry, SW_data.Rz, _, _ = SM_to_SMD(
                SW_data.Rx,
                SW_data.Ry,
                SW_data.Rz,
                arr_time(SW_data.index),
                gorgonops=gorgonops,
            )
        if coords != "sim":
            SW_data.vx, SW_data.vy, SW_data.Bx, SW_data.By, SW_data.Rx, SW_data.Ry = (
                -SW_data.vx,
                -SW_data.vy,
                -SW_data.Bx,
                -SW_data.By,
                -SW_data.Rx,
                -SW_data.Ry,
            )

        if mean_Bx:
            SW_data.Bx = 0 * SW_data.Bx + np.mean(
                SW_data.Bx
            )  # Use average Bx during chosen period (Bx must be time-constant)
            sim_params["Bx"] = np.mean(SW_data.Bx) * 1e-9
        else:
            SW_data.Bx = 0 * SW_data.Bx
            sim_params["Bx"] = 0.0

        def inflow_limit(vx, vy, max_ang):
            return np.where(
                abs(vy) > abs(vx) * np.tan(max_ang * np.pi / 180),
                abs(vx) * np.tan(max_ang * np.pi / 180) * np.sign(vy),
                vy,
            )

        if max_yz_inflow[0] is not None:
            SW_data.vy = inflow_limit(SW_data.vx, SW_data.vy, max_yz_inflow[0])
        if max_yz_inflow[1] is not None:
            SW_data.vz = inflow_limit(SW_data.vx, SW_data.vz, max_yz_inflow[1])

        if use_SMD:
            _, _, _, sim_params["Mdir_theta"], SW_data.tilt = SM_to_SMD(
                M_x, M_y, M_z, arr_time(SW_data.index), gorgonops=gorgonops
            )
        else:
            sim_params["Mdir_theta"] = 0.0

        from io import BytesIO
        from pkgutil import get_data

        data = get_data(__name__, "../geomagnetic/data/IGRF_poles.csv")
        poles = pd.read_csv(BytesIO(data), encoding="utf8")
        poles["Times"] = [dt.datetime(int(i), 7, 2, 0, 0) for i in poles["YEAR"]]
        poles["DIP_MOM"] = [i * 1e22 for i in poles["DIP_MOM"]]
        poles.set_index("Times", inplace=True)
        poles = poles["DIP_MOM"]
        poles = poles.resample("1M").mean().interpolate(method="linear")

        sim_params["M"] = poles.iloc[
            poles.index.get_indexer(
                [pd.to_datetime(arr_time(SW_data.index)[0])], method="nearest"
            )
        ]

        if init_data is not None:
            SW_data = init_transition(
                init_data, SW_data
            )  # linearly ramp up data from initialized state for given period in
            # simulation coords

        SW_Input = np.array(
            [
                (SW_data.index - SW_data.index[0]).total_seconds() + simtime,
                SW_data.tilt,
                SW_data.Rx,
                SW_data.Ry,
                SW_data.Rz,
                SW_data.n * proton_mass * 1e6,
                SW_data.Ti,
                SW_data.Ti,
                SW_data.vx * 1e3,
                SW_data.vy * 1e3,
                SW_data.vz * 1e3,
                SW_data.Bx * 1e-9,
                SW_data.By * 1e-9,
                SW_data.Bz * 1e-9,
            ]
        ).T
        if os.path.exists(out_dir + fnm) and SW_append is True:
            with open(out_dir + fnm, "ab") as f:
                np.savetxt(f, SW_Input, delimiter=",", fmt="% 1.5e")
        else:
            np.savetxt(out_dir + fnm, SW_Input, delimiter=",", fmt="% 1.5e")

        sim_params[
            "F_10.7"
        ] = 100.0  # Solar 10.7 cm solar radio flux (leaving here for future utility)

    if not write:
        SW_data = None
        SW_Input = None

    return SW_data_out, SW_data, SW_Input, sim_params


def read_SW_input(file, starttime, coords="sim"):
    """Read in existing Gorgon solar wind data input file for analysis and plotting.

    Args:
    ----
        file (str): Path to input data file (by default called SW_Input.dat)
        starttime (datetime): Date and time in UT corresponding to t = 0 in the
        simulation.
        coords (str, optional): Coordinate system out of 'GSE', 'GSM', 'SM' and 'sim'
        in which to return data. Defaults to 'sim'.

    Returns:
    -------
        pd.DataFrame: Pandas dataframe containing resulting data.

    """

    def arr_time(index):
        return np.array([t for t in index])

    SW_data_in = np.loadtxt(file, delimiter=",").astype(float)
    SW_data = pd.DataFrame(SW_data_in[:, 1:])
    SW_data.columns = [
        "tilt",
        "Rx",
        "Ry",
        "Rz",
        "rho",
        "Ti",
        "Te",
        "vx",
        "vy",
        "vz",
        "Bx",
        "By",
        "Bz",
    ]
    SW_data.index = [starttime + dt.timedelta(seconds=s) for s in SW_data_in[:, 0]]

    for i in ["x", "y", "z"]:
        SW_data["B" + i] /= 1e-9
        SW_data["v" + i] /= 1e3
    SW_data["n"] = SW_data["rho"] / (1.67e-27 * 1e6)
    SW_data["B"] = np.sqrt(SW_data["Bx"] ** 2 + SW_data["By"] ** 2 + SW_data["Bz"] ** 2)
    SW_data["v"] = np.sqrt(SW_data["vx"] ** 2 + SW_data["vy"] ** 2 + SW_data["vz"] ** 2)

    if coords == "SM":
        SW_data["Bx"], SW_data["By"] = -SW_data["Bx"], -SW_data["By"]
        SW_data["vx"], SW_data["vy"] = -SW_data["vx"], -SW_data["vy"]
    elif coords == "GSM":
        SW_data["Bx"], SW_data["By"], SW_data["Bz"] = GSM_to_SM(
            -SW_data["Bx"],
            -SW_data["By"],
            SW_data["Bz"],
            arr_time(SW_data.index),
            inv=True,
        )
        SW_data["vx"], SW_data["vy"], SW_data["vz"] = GSM_to_SM(
            -SW_data["vx"],
            -SW_data["vy"],
            SW_data["vz"],
            arr_time(SW_data.index),
            inv=True,
        )
    elif coords == "GSE":
        SW_data["Bx"], SW_data["By"], SW_data["Bz"] = GSE_to_SM(
            -SW_data["Bx"],
            -SW_data["By"],
            SW_data["Bz"],
            arr_time(SW_data.index),
            inv=True,
        )
        SW_data["vx"], SW_data["vy"], SW_data["vz"] = GSE_to_SM(
            -SW_data["vx"],
            -SW_data["vy"],
            SW_data["vz"],
            arr_time(SW_data.index),
            inv=True,
        )

    return SW_data
