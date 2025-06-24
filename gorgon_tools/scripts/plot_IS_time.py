"""Todo: Docstring for plot_IS_time."""
import datetime as dt
import sys
import warnings

from ..ionosphere.IS_import import ionosphere
from ..ionosphere.visualise import plot_GEO, plot_polar

warnings.filterwarnings("ignore")

path_to_data = sys.argv[1]  # Directory containing simulation data
iono = ionosphere(path_to_data)  # Ionosphere class
time = sys.argv[2]  # chosen time in seconds to plot

try:
    dt_str_format_in = "%Y-%m-%d_%H:%M:%S"  # Format of datettime read in by script
    dt_str_format_out = (
        "%Y-%m-%d_%H_%M"  # Format of datettime used for output file names
    )
    t0_UT = dt.datetime.strptime(sys.argv[3], dt_str_format_in)

    time_UT = t0_UT + dt.timedelta(seconds=int(time))  # UT time at timestep
    iono.datetime = time_UT
    time_str = time_UT.strftime(dt_str_format_out)
except Exception as e:
    t0_UT = None
    iono.datetime = None
    time_str = str(time)
    print(e)

# Import ionospheric data at timestep
iono.import_timestep(iono.timestep(int(time)))

# Polar (SM) plots
plt_list = [
    [
        "FAC",
        {
            "name": r"$j_{||}$",
            "unit": r"$\mu$Am$^{-2}$",
            "norm": 1e-6,
            "min": -1,
            "max": 1,
            "cmap": "RdBu_r",
            "contours": True,
        },
    ],
    [
        "phi",
        {
            "name": r"$\phi$",
            "unit": r"kV",
            "norm": 1e3,
            "min": -80,
            "max": 80,
            "cmap": "viridis",
            "contours": True,
        },
    ],
    [
        "sig_P",
        {
            "name": r"$\Sigma_P$",
            "unit": r"mho",
            "norm": 1,
            "min": 0,
            "max": 20,
            "cmap": "YlGn",
            "contours": True,
        },
    ],
    [
        "sig_H",
        {
            "name": r"$\Sigma_H$",
            "unit": r"mho",
            "norm": 1,
            "min": 0,
            "max": 20,
            "cmap": "YlGn",
            "contours": True,
        },
    ],
]
for plts in plt_list:
    plot_polar(
        iono,
        plts,
        t0_UT=t0_UT,
        pcolormesh=False,
        disp=False,
        filename=path_to_data + "/Plots/IS/" + plts[0] + "_" + time_str,
        fileformat="jpg",
    )

# Geographic plots
if t0_UT is not None:
    plt_list = [
        [
            "jvec",
            {
                "name": r"$j_{\perp}$",
                "comp": "mag",
                "unit": r"Am$^{-1}$",
                "norm": 1,
                "min": 0.0,
                "cmap": "Greens",
                "contours": False,
            },
        ],
        [
            "JH",
            {
                "name": r"Joule Heating",
                "unit": r"mWm$^{-2}$",
                "norm": 1e-3,
                "min": 0.0,
                "cmap": "hot",
                "contours": False,
            },
        ],
    ]
    for plts in plt_list:
        plot_GEO(
            iono,
            plts,
            t0_UT,
            region="Global",
            pcolormesh=False,
            disp=False,
            filename=path_to_data + "/Plots/IS/" + plts[0] + "_Global_" + time_str,
            fileformat="jpg",
        )
        plot_GEO(
            iono,
            plts,
            t0_UT,
            region="Europe",
            pcolormesh=False,
            disp=False,
            filename=path_to_data + "/Plots/IS/" + plts[0] + "_Europe_" + time_str,
            fileformat="jpg",
        )
        plot_GEO(
            iono,
            plts,
            t0_UT,
            region="UK",
            pcolormesh=False,
            disp=False,
            filename=path_to_data + "/Plots/IS/" + plts[0] + "_UK_" + time_str,
            fileformat="jpg",
        )
