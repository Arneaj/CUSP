import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, spherical_to_cartesian, Me25_leaky, delete_last_line
from earth_pos_detection import get_earth_pos
from streamlines_3d import find_streamline, smooth_interpolation


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

B = import_from(f"{filepath}/B_processed_real.txt")
B_norm = np.linalg.norm( B, axis=3 )

J_norm = None
V = None

earth_pos = get_earth_pos( B_norm )
print( "earth pos = ", earth_pos )

shape = B_norm.shape
print( "shape = ", B_norm.shape )

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )



def streamlines_casting():
    theta, phi = np.meshgrid( np.linspace( -np.pi*0.9, np.pi*0.9, 50 ), np.linspace( -np.pi*0.5, np.pi*0.5, 10 ), indexing='ij' )
    R = Me25_leaky( params, theta, phi )

    X,Y,Z = spherical_to_cartesian( R, theta, phi, earth_pos )
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    streamlines = []
    
    for i in range(len(X)):
        streamline = find_streamline( B[:,:,:,0], B[:,:,:,1], B[:,:,:,2], X[i], Y[i], Z[i], 0.5 )
        streamline = smooth_interpolation(streamline, 5)
        streamlines.append(streamline)
        
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(0, B.shape[0])
    ax.set_ylim(0, B.shape[1])
    ax.set_zlim(0, B.shape[2])
    
    ax.view_init(elev=20., azim=150)
    
    for streamline in streamlines:        
        ax.plot( streamline[:,0], streamline[:,1], streamline[:,2], color=(0.1,0.1,0.1,0.2) )
    
    ax.set_xlabel("$x$ [$R_E$]")
    ax.set_ylabel("$y$ [$R_E$]")
    ax.set_zlabel("$z$ [$R_E$]")
    
    plt.savefig("streamlines_from_MP.svg")
    
    print("nb_streamlines = ", len(streamlines))


if __name__ == "__main__":
    streamlines_casting()

