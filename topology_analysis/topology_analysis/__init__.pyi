"""
Topology Analysis
-----------------------

.. currentmodule:: topology_analysis

.. autosummary::
    :toctree: _generate
    
    preprocess
"""

import numpy as np
from numpy.typing import NDArray

def preprocess( matrix: NDArray[np.float64], X: NDArray[np.float64], Y: NDArray[np.float64], Z: NDArray[np.float64], new_shape: NDArray[np.float64] ) -> NDArray[np.float64]:
    """
    Transform a matrix from a non uniform grid to a uniform grid,
    using X, Y and Z containing the actual position of the center of each grid of each matrix indices. 
    """
