import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, Me25_leaky, spherical_to_cartesian

import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
saturation = 3e-9


theta = np.linspace(0, np.pi*0.99, 100)

earth_pos = [30.75, 58, 58]

fig, axes = plt.subplots(1, 2)

fig.set_figwidth(10)
fig.set_figheight(4)

############### ME25

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )


# TODO: THIS IS SPECIFICALLY FOR THE Me25_leaky FUNCTION
Theta = np.linspace(-np.pi*0.99, np.pi*0.99, 200)
Phi = 0
R = Me25_leaky( params, Theta, Phi )
X1, _, Z1 = spherical_to_cartesian( R, Theta, Phi, earth_pos )

Phi = np.pi/2
R = Me25_leaky( params, Theta, Phi )
X2, Y2, _ = spherical_to_cartesian( R, Theta, Phi, earth_pos )


### vmax
# B: 1e-7
# J: 5e-10
# V: None


index = 58


J_xy = axes[0].imshow(J_norm[::-1,:,index], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xy, ax=axes[0])
J_xy = axes[0].plot(Y2, J_norm.shape[0] - X2)
axes[0].set_title(fr"$||J|| \in ({index},\hat x, \hat y)$")
axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")
axes[0].set_xlim(0, J_norm.shape[1]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)

J_xz = axes[1].imshow(J_norm[::-1,index,:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xz, ax=axes[1])
J_xy = axes[1].plot(Z1, J_norm.shape[0] - X1)
axes[1].set_title(fr"$||J|| \in ({index},\hat x, \hat z)$")
axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")
axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)



plt.savefig("../images/line_guess.svg")
