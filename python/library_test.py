import numpy as np
import matplotlib.pyplot as plt
import topology_analysis as ta
import gorgon
import time

import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

Rho = gorgon.import_from_bin( filepath + "/Rho_processed_real.bin" )
earth_pos = np.array( [30, 58, 58], dtype=np.float32 )

t0 = time.time()
bs_radius = ta.get_bowshock_radius( 0.0, 0.0, Rho, earth_pos, 0.1 )
t1 = time.time()
print(f"Finished in {t1-t0}s -> Bowshock radius for (theta,phi) = (0.0, 0.0):", bs_radius)

t0 = time.time()
BS = ta.get_bowshock( Rho, earth_pos, 0.1, 50, 50 )
t1 = time.time()
print(f"Finished in {t1-t0}s -> Found entire Bowshock")
# print("All bowshock radii:", BS)

J_norm = gorgon.import_from_bin( filepath + "/J_norm_processed_real.bin" )

X, Y, Z = gorgon.spherical_to_cartesian( BS[:,2], BS[:,0], BS[:,1], earth_pos )

is_in_plane = np.abs(Y-58) < 2

X_plot = X[is_in_plane]
Z_plot = Z[is_in_plane]

plt.imshow( Rho[:,58,:], cmap="inferno", norm="log" )
plt.scatter( Z_plot, X_plot, s=1.0 )
plt.savefig( "../images/bowshock.svg" )
