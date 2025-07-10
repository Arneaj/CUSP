import numpy as np
import matplotlib.pyplot as plt

from gorgon import Me25, import_from, Me25_leaky

import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
# J_norm = np.linalg.norm( B, axis=3 )
saturation = 3e-9


theta = np.linspace(0, np.pi*0.99, 100)

fig, axes = plt.subplots( 1, 2 )


earth_pos = [30.75, 58, 58]

############## LIN10

r_0 = 10.4882

alpha_0 = 0.5776
alpha_1 = 0
alpha_2 = 0.1404
e = 0

d = 1.846
l = 1.1574
s = 0.3771

r1 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d, l, s, d, l, s, e], theta, 0 ) )
r2 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d, l, s, d, l, s, e], theta, np.pi ) )

X1 = r1 * np.cos(theta)
Z1 = r1 * np.sin(theta)

X2 = r1 * np.cos(theta)
Z2 = r1 * np.sin(theta)

X = np.concatenate( [X2[::-1], X1] )
Z = np.concatenate( [-Z2[::-1], Z1] )

X += J_norm.shape[0] - earth_pos[0]
Z += earth_pos[2]


axes[0].imshow( J_norm[::-1,58,:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none" )
axes[0].plot( Z, X )
axes[0].set_title( r"Gorgon data VS Liu12 model" )
axes[0].set_xlim(0, J_norm.shape[2]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)

############### ME25

theta = np.linspace(-np.pi*0.99, np.pi*0.99, 200)

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )

r1 = ( Me25_leaky( params, theta, 0 ) )
# r2 = ( Me25_leaky( params, theta, np.pi ) )

X1 = r1 * np.cos(theta)
Z1 = r1 * np.sin(theta)

# X2 = r1 * np.cos(theta)
# Z2 = r1 * np.sin(theta)

# X = np.concatenate( [X2[::-1], X1] )
# Z = np.concatenate( [-Z2[::-1], Z1] )

X = X1
Z = Z1

X += J_norm.shape[0] - earth_pos[0]
Z += earth_pos[2]


axes[1].imshow( J_norm[::-1,58,:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none" )
axes[1].plot( Z, X )
axes[1].set_title( r"Gorgon data VS Me25 model" )
axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)


fig.suptitle(filepath)


plt.savefig("../images/gorgon_vs_liu12.svg")
