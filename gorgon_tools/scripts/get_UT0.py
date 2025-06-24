"""Get the UT time at the start of the simulation."""
import datetime as dt
import sys


def get_UT0(t0, init_offset=7200):
    """Get the UT time at the start of the simulation.

    Compute the UT time at the start of the simulation, given a specified ingestion
    start time and initialization offset.

    Args:
    ----
        t0 (str): The ingestion start time in appropriate format ("%Y-%m-%d_%H:%M:%S").
        init_offset (int, optional): The initialization offset in seconds.
        Defaults to 7200.

    Returns:
    -------
        str: The UT time at the start of the simulation in appropriate format
        ("%Y-%m-%d_%H:%M:%S").

    """
    UT0 = dt.datetime.strptime(t0, "%Y-%m-%d_%H:%M:%S") - dt.timedelta(
        seconds=int(init_offset)
    )
    return UT0.strftime("%Y-%m-%d_%H:%M:%S")


if __name__ == "__main__":
    # create help message for command line arguments
    if len(sys.argv) == 1:
        print("Usage: python get_UT0.py <YYYY-MM-DD_HH:MM:SS> <optional: init_offset>")
        sys.exit()
    elif len(sys.argv) > 2:
        UT0 = get_UT0(sys.argv[1], float(sys.argv[2]))
        print(UT0)
    else:
        UT0 = get_UT0(sys.argv[1])
        print(UT0)
