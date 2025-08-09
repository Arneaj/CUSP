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

if len(sys.argv) < 3 or sys.argv[2] == "xz":
    axis = 'xz'
elif sys.argv[2] == "xy":
    axis = 'xy'
else:
    print( "Please provide xy or xz" )

Rho = gorgon.import_from_bin( filepath + "/Rho_processed_real.bin" )
earth_pos = 2.0*np.array( [30, 58, 58], dtype=np.float32 )

t0 = time.time()
bs_radius = ta.get_bowshock_radius( 0.0, 0.0, Rho, earth_pos, 0.1 )
t1 = time.time()
print(f"Finished in {t1-t0}s -> Bowshock radius for (theta,phi) = (0.0, 0.0):", bs_radius)

t0 = time.time()
BS = ta.get_bowshock( Rho, earth_pos, 0.1, 4, 50 )
t1 = time.time()
print(f"Finished in {t1-t0}s -> Found entire Bowshock")

J_norm = gorgon.import_from_bin( filepath + "/J_norm_processed_real.bin" )

t0 = time.time()
MP = ta.get_interest_points(
    J_norm, earth_pos, 
    Rho,
    0.0, np.pi*0.9,  
    50, 4,
    0.1, 0.1,
    0.4, 0.6, 4,
    1.5, 3.0, 20
)
t1 = time.time()
print(f"Finished in {t1-t0}s -> Found entire Magnetopause")

X_BS, Y_BS, Z_BS = gorgon.spherical_to_cartesian( BS[:,2], BS[:,0], BS[:,1], earth_pos )
X_MP, Y_MP, Z_MP = gorgon.spherical_to_cartesian( MP[:,2], MP[:,0], MP[:,1], earth_pos )

if axis == "xz":
    is_in_plane_BS = np.abs(Y_BS-int(earth_pos[1])) < 1

    X_BS_plot = X_BS[is_in_plane_BS]
    Z_BS_plot = Z_BS[is_in_plane_BS]

    is_in_plane_MP = np.abs(Y_MP-int(earth_pos[1])) < 1

    X_MP_plot = X_MP[is_in_plane_MP]
    Z_MP_plot = Z_MP[is_in_plane_MP]

    # plt.imshow( np.moveaxis(Rho[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    plt.imshow( np.moveaxis(J_norm[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
    plt.colorbar()
    plt.scatter( X_BS_plot, Z_BS_plot, s=1.0 )
    plt.scatter( X_MP_plot, Z_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
    
    plt.ylabel(r"$z \in [-58; 58] R_E$")
    
else:
    is_in_plane_BS = np.abs(Z_BS-int(earth_pos[2])) < 1

    X_BS_plot = X_BS[is_in_plane_BS]
    Y_BS_plot = Y_BS[is_in_plane_BS]

    is_in_plane_MP = np.abs(Z_MP-int(earth_pos[2])) < 1

    X_MP_plot = X_MP[is_in_plane_MP]
    Y_MP_plot = Y_MP[is_in_plane_MP]

    # plt.imshow( np.moveaxis(Rho[:,int(earth_pos[2]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    plt.imshow( np.moveaxis(J_norm[:,:,int(earth_pos[2])], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
    plt.colorbar()
    plt.scatter( X_BS_plot, Y_BS_plot, s=1.0 )
    plt.scatter( X_MP_plot, Y_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
    
    plt.ylabel(r"$y \in [-58; 58] R_E$")

plt.xlabel(r"$x \in [-30; 128] R_E$")


plt.savefig( "../images/bowshock.svg" )
