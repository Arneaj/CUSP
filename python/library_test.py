import time

t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import topology_analysis as ta
from gorgon_tools.magnetosphere import gorgon_import
import gorgon
import sys
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Modules loaded")

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

# /rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1
filepath = sys.argv[1]

if len(sys.argv) < 3:
    print("No Timestep path given!")
    exit(1)

timestep = sys.argv[2]


if len(sys.argv) < 4 or sys.argv[3] == "xz":
    axis = 'xz'
elif sys.argv[3] == "xy":
    axis = 'xy'
else:
    print( "Please provide xy or xz" )




t0 = time.time()
sim = gorgon_import.gorgon_sim(data_dir=filepath)
index_of_timestep = np.where( sim.times == float(timestep) )[0][0]
sim.import_timestep(index_of_timestep)
sim.import_space( filepath + "/MS/x00_Bvec_c-" + timestep + ".pvtr")

Rho: np.ndarray = sim.arr["rho"]
J: np.ndarray = sim.arr["jvec"]

X: np.ndarray = sim.xc; Y: np.ndarray = sim.yc; Z: np.ndarray = sim.zc
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Files read")


print( "J dtype: ", J.dtype )
print( "J shape: ", J.shape )
print( "J strides: ", J.strides )

    





t0 = time.time()
extra_precision = 2.0

shape_realx2 = np.array([
    int( extra_precision * (X[-1]-X[0]) ), 
    int( extra_precision * (Y[-1]-Y[0]) ), 
    int( extra_precision * (Z[-1]-Z[0]) ),
    3
], dtype=np.int16)

J_norm = np.linalg.norm( J, axis=3 )

J_processed: np.ndarray = ta.preprocess( J, X, Y, Z, shape_realx2 )
Rho_processed: np.ndarray = ta.preprocess( Rho, X, Y, Z, shape_realx2 )
J_norm_processed: np.ndarray = ta.preprocess( J_norm, X, Y, Z, shape_realx2 )

# J_norm_processed = np.linalg.norm( J_processed, axis=3 )

earth_pos = extra_precision * np.array( [30, 58, 58], dtype=np.float32 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Preprocessing done")


print( "J_processed dtype: ", J_processed.dtype )
print( "J_processed shape: ", J_processed.shape )
print( "J_processed strides: ", J_processed.strides )



t0 = time.time()
bs_radius = ta.get_bowshock_radius( 0.0, 0.0, Rho_processed, earth_pos, 0.1 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Bowshock radius for (theta,phi) = (0.0, 0.0):", bs_radius)





t0 = time.time()
BS = ta.get_bowshock( Rho_processed, earth_pos, 0.1, 4, 50 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Found entire Bowshock")




t0 = time.time()
MP = ta.get_interest_points(
    J_norm_processed, earth_pos, 
    Rho_processed,
    0.0, np.pi*0.9,  
    50, 4,
    0.1, 0.1,
    0.4, 0.6, 4,
    1.5, 3.0, 20
)
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Found entire Magnetopause")







X_BS, Y_BS, Z_BS = gorgon.spherical_to_cartesian( BS[:,2], BS[:,0], BS[:,1], earth_pos )
X_MP, Y_MP, Z_MP = gorgon.spherical_to_cartesian( MP[:,2], MP[:,0], MP[:,1], earth_pos )

if axis == "xz":
    is_in_plane_BS = np.abs(Y_BS-int(earth_pos[1])) < 1

    X_BS_plot = X_BS[is_in_plane_BS]
    Z_BS_plot = Z_BS[is_in_plane_BS]

    is_in_plane_MP = np.abs(Y_MP-int(earth_pos[1])) < 1

    X_MP_plot = X_MP[is_in_plane_MP]
    Z_MP_plot = Z_MP[is_in_plane_MP]

    # plt.imshow( np.moveaxis(Rho_processed[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    plt.imshow( np.moveaxis(J_norm_processed[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
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

    # plt.imshow( np.moveaxis(Rho_processed[:,int(earth_pos[2]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    plt.imshow( np.moveaxis(J_norm_processed[:,:,int(earth_pos[2])], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
    plt.colorbar()
    plt.scatter( X_BS_plot, Y_BS_plot, s=1.0 )
    plt.scatter( X_MP_plot, Y_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
    
    plt.ylabel(r"$y \in [-58; 58] R_E$")

plt.xlabel(r"$x \in [-30; 128] R_E$")


plt.savefig( "../images/bowshock.svg" )
