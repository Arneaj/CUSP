import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, Me25_fix, spherical_to_cartesian

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
fig.set_figheight(5.5)

############### ME25

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )


Theta = np.linspace(0, np.pi*0.99, 200)
Phi = 0
R1 = Me25_fix( params, Theta, Phi )
X11, _, Z11 = spherical_to_cartesian( R1, Theta, Phi, earth_pos )
Phi = np.pi
R1 = Me25_fix( params, Theta, Phi )
X12, _, Z12 = spherical_to_cartesian( R1, Theta, Phi, earth_pos )
X1 = np.concatenate( [X12[::-1], X11] )
Z1 = np.concatenate( [Z12[::-1], Z11] )

Theta = np.linspace(0, np.pi*0.99, 200)
Phi = np.pi/2
R1 = Me25_fix( params, Theta, Phi )
X21, Y21, _ = spherical_to_cartesian( R1, Theta, Phi, earth_pos )
Phi = -np.pi/2
R1 = Me25_fix( params, Theta, Phi )
X22, Y22, _ = spherical_to_cartesian( R1, Theta, Phi, earth_pos )
X1 = np.concatenate( [X22[::-1], X21] )
Y1 = np.concatenate( [Y22[::-1], Y21] )


### vmax
# B: 1e-7
# J: 5e-10
# V: None


index = 58


J_xy = axes[0].imshow(J_norm[::-1,:,index], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xy, ax=axes[0], shrink=1)
J_xy = axes[0].plot(Y2, J_norm.shape[0] - X2)
axes[0].set_title(fr"$||J|| \in ({index},\hat x, \hat y)$")
axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")
axes[0].set_xlim(0, J_norm.shape[1]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)

J_xz = axes[1].imshow(J_norm[::-1,index,:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xz, ax=axes[1], shrink=1)
J_xy = axes[1].plot(Z1, J_norm.shape[0] - X1)
axes[1].set_title(fr"$||J|| \in ({index},\hat x, \hat z)$")
axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")
axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)


name_of_datapoint = filepath.split("/")[-1]

run_nb, timestep = name_of_datapoint.split("_")


fig.suptitle(f"{run_nb} at timestep $t =$ {timestep}s")


plt.savefig("../images/line_guess.svg")
