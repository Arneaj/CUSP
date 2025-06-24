"""Module for the fieldlines class.

Created on Wed Sep  6 14:14:46 2017

@author: Lars
"""
import numpy as np

from .streamline import streamline_array


class fieldlines(streamline_array):
    """Class for storing and manipulating fieldlines."""

    def __init__(self, n_steps, step_size, inner_boundary=False):
        """Initialise the fieldlines class."""
        streamline_array.__init__(
            self, n_steps, step_size, direction=0, inner_boundary=inner_boundary
        )

    def calc_linkage(self, sim):
        """Calculate the linkage of the field lines based on their footpoints.

        Args:
        ----
            sim: Simulation object containing the necessary data.

        Returns:
        -------
            None

        """

        def end_cat(ROT):
            f, r = ROT

            if r == 2 and f == 2:
                link = 1  # Solar Wind
            elif r == 3 and f == 3:
                link = 2  # Closed
            elif r == 3 and f == 2:
                link = 3  # North-Open
            elif r == 2 and f == 3:
                link = 4  # South-Open
            elif r == 2 or f == 2:
                link = 5  # SW-Inc
            elif r == 3:
                link = 6  # North-Inc
            elif f == 3:
                link = 7  # South-Inc
            else:
                link = 8  # Inc-Inc

            return link

        self.cell_data["link"] = np.array([end_cat(ROTi) for ROTi in self.ROT])

    def filter(self, i):
        """Filter the field line data based on the given indices.

        Args:
        ----
            i (list): List of indices to keep.

        Returns:
        -------
            None

        """
        self.xs = self.xs[i]
        self.ROT = self.ROT[i]

        self.ns = self.ns[i]

        for s in self.var:
            self.var[s] = self.var[s][i]

        # self.x0 = self.x0[i]

        for s in self.cell_data:
            self.cell_data[s] = self.cell_data[s][i]
