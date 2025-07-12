import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, spherical_to_cartesian, Me25_fix, delete_last_line
from earth_pos_detection import get_earth_pos
from streamlines_3d import find_streamline, smooth_interpolation


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)
    
if len(sys.argv) < 3:
    print("No vector given!")
    exit(1)

filepath = sys.argv[1]

vec_used = sys.argv[2]

B = import_from(f"{filepath}/B_processed_real.txt")
B_norm = np.linalg.norm( B, axis=3 )

earth_pos = get_earth_pos( B_norm )
print( "earth pos = ", earth_pos )

shape = B_norm.shape
print( "shape = ", B_norm.shape )

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )


def streamlines_casting():   
    if vec_used == "B":
        vec = B
    if vec_used == "V":
        vec = import_from(f"{filepath}/V_processed_real.txt")
    else:
        print("incorrect vector!")
        exit(1)
     
    nb_theta = 60
    nb_phi = 15
     
    theta, phi = np.meshgrid( np.linspace( 0, np.pi*0.9, nb_theta ), np.linspace( -np.pi, np.pi, nb_phi ), indexing='ij' )
    R = Me25_fix( params, theta, phi )

    X,Y,Z = spherical_to_cartesian( R, theta, phi, earth_pos )
    
    B_thing = B[np.array(X, dtype=np.int16), np.array(Y, dtype=np.int16), np.array(Z, dtype=np.int16)]
    B_thing_norm = np.linalg.norm( B_thing, axis=2 )
    
    saturation = 0.4
    B_thing_norm = B_thing_norm * (B_thing_norm < np.max(B_thing_norm)*saturation) + np.max(B_thing_norm)*saturation * (B_thing_norm >= np.max(B_thing_norm)*saturation)
    
    color = (B_thing_norm - np.min(B_thing_norm)) / (np.max(B_thing_norm) - np.min(B_thing_norm))
    cmap = plt.get_cmap("inferno")
    color = cmap( color )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel=r"$x \in [-30; 128] R_E$", ylabel=r"$y \in [-58; 58] R_E$", zlabel=r"$z \in [-58; 58] R_E$")
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    ax.view_init(elev=0, azim=270)
    ax.figure.subplots_adjust(0, 0, 1, 1)

    ax.plot_surface(X, Y, Z, facecolors=color, shade=False)
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    X_mask = X>earth_pos[0]
    X = X[X_mask]
    Y = Y[X_mask]
    Z = Z[X_mask]
    
    streamlines = []
    
    for i in range(len(X)):
        streamline = find_streamline( vec[:,:,:,0], vec[:,:,:,1], vec[:,:,:,2], X[i], Y[i], Z[i], 0.5 )
        streamline = smooth_interpolation(streamline, 5)
        streamlines.append(streamline)
    
    for streamline in streamlines:        
        ax.plot( streamline[:,0], streamline[:,1], streamline[:,2], color=(0.1,0.1,0.1,0.2) )
    
    plt.savefig("../images/streamlines_from_MP.svg")


if __name__ == "__main__":
    streamlines_casting()

