import numpy as np
import matplotlib.pyplot as plt
import topology_analysis as ta
import gorgon

Rho = gorgon.import_from_bin( "../data/Run1_23100/Rho_processed_real.bin" )
P_E = ta.Point(30, 58, 58)
earth_pos = [30, 58, 58]

bs_radius = ta.get_bowshock_radius( 0.0, 0.0, Rho, P_E, 0.1 )
print("Bowshock radius for (theta,phi) = (0.0, 0.0):", bs_radius)

BS = ta.get_bowshock( Rho, P_E, 0.1, 50, 50 )
# print("All bowshock radii:", BS)

J_norm = gorgon.import_from_bin( "../data/Run1_23100/J_norm_processed_real.bin" )

X, Y, Z = gorgon.spherical_to_cartesian( BS[:,2], BS[:,0], BS[:,1], earth_pos )

is_in_plane = np.abs(Y-58) < 2

X_plot = X[is_in_plane]
Z_plot = Z[is_in_plane]

plt.imshow( J_norm[:,58,:], cmap="inferno", norm="log" )
plt.scatter( Z_plot, X_plot, s=1.0 )
plt.savefig( "../images/bowshock.svg" )
