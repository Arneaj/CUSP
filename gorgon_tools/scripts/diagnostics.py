"""Todo: Docstring for diagnostics."""
import sys

import matplotlib.pyplot as plt

from ..geomagnetic.visualise import plot_B
from ..magnetosphere.gorgon_import import gorgon_sim
from .gorgon_report_functions import (
    import_dB_metrics,
    import_GSO_vars,
    import_IS_metrics,
    import_MS_metrics,
    import_xy,
    plot_energy,
    plot_ground_metrics,
    plot_GSO_vars,
    plot_IS_metrics,
    plot_MS_metrics,
    plot_quick_look,
    plot_real_time_comparision,
    plot_SW_metrics,
    plot_timestep,
    plot_work,
)

path_to_data = sys.argv[1]
sim = gorgon_sim(path_to_data)

t_fun = [lambda t: t / 60, "(min)"]

try:
    xy = import_xy(path_to_data + "/xy" + sim.index[1:] + ".csv")

    fig, ax = plot_quick_look(xy, t_fun)
    plt.savefig(path_to_data + "/Plots/Diagnostics/Summary.jpg", bbox_inches="tight")
    plt.close()

    plot_energy(xy, t_fun)
    plt.savefig(path_to_data + "/Plots/Diagnostics/Energy.jpg", bbox_inches="tight")
    plt.close()

    plot_work(xy, t_fun)
    plt.savefig(path_to_data + "/Plots/Diagnostics/Work.jpg", bbox_inches="tight")
    plt.close()

    plot_timestep(xy, t_fun)
    plt.savefig(path_to_data + "/Plots/Diagnostics/Timestep.jpg", bbox_inches="tight")
    plt.close()

    plot_real_time_comparision(
        path_to_data, xy, t_fun, prefix=sim.index[1:], include_xy=False
    )
    plt.savefig(path_to_data + "/Plots/Diagnostics/Runtime.jpg", bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Failed to generate MHD diagnostics:", str(e))


try:
    MS = import_MS_metrics(path_to_data + "/MS_Vars.csv")
    plot_MS_metrics(MS, t_fun)
    plt.savefig(path_to_data + "/Plots/MS_Params.jpg", bbox_inches="tight")
    plt.close()

    plot_SW_metrics(MS, t_fun)
    plt.savefig(path_to_data + "/Plots/SW_Params.jpg", bbox_inches="tight")
    plt.close()

    GSO = import_GSO_vars(path_to_data, t_fun)
    fig, ax = plot_GSO_vars(GSO, t_fun)
    plt.savefig(path_to_data + "/Plots/GSO_Params.jpg", bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Failed to generate magnetospheric metrics.", str(e))

try:
    IS = import_IS_metrics(path_to_data + "/IS_Vars.csv")
    plot_IS_metrics(IS, t_fun)
    plt.savefig(path_to_data + "/Plots/IS_Params.jpg", bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Failed to generate ionospheric metrics.", str(e))

try:
    dB, inds = import_dB_metrics(path_to_data, t_fun)
    fig, ax = plot_ground_metrics(inds, t_fun)
    plt.savefig(path_to_data + "/Plots/dB_Params.jpg", bbox_inches="tight")
    plt.close()

    for stn in dB.stations:
        plot_B(
            dB,
            [stn],
            disp=False,
            filename=path_to_data + "/Plots/dB/" + stn + ".jpg",
            fileformat="jpg",
        )
except Exception as e:
    print("Failed to generate ground metrics.", str(e))
