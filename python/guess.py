import numpy as np
import matplotlib.pyplot as plt

from gorgon import Me25, import_from

import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
saturation = 3e-9


theta = np.linspace(0, np.pi*0.99, 100)

earth_pos = [29.75, 58, 58]

fig, axes = plt.subplots(1, 2)

fig.set_figwidth(10)
fig.set_figheight(4)

############### ME25

with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )


X, Y, Z = np.meshgrid(
    np.arange( J_norm.shape[0] ),
    np.arange( J_norm.shape[1] ),
    np.arange( J_norm.shape[2] ),
    indexing='ij'
)

X = earth_pos[0] - X
Y -= earth_pos[1]
Z -= earth_pos[2]

R = np.sqrt( X*X + Y*Y + Z*Z )
Theta = np.arccos( X / np.maximum(1, R) )
Phi = np.arccos( Z / np.maximum(1, np.sqrt( Y*Y + Z*Z )) )
Phi = Phi * (Y>0) - Phi*(Y<=0)
predictedR = Me25( params, Theta, Phi )

Mask = R <= predictedR


### vmax
# B: 1e-7
# J: 5e-10
# V: None


index = 58


J_xy = axes[0].imshow(J_norm[::-1,:,index], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xy, ax=axes[0])
J_xy = axes[0].imshow(np.ones_like(J_norm[::-1,:,index]), alpha=0.8*Mask[::-1,:,index], cmap="Paired")
axes[0].set_title(fr"$||J|| \in ({index},\hat x, \hat y)$")
axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")
axes[0].set_xlim(0, J_norm.shape[1]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)

J_xz = axes[1].imshow(J_norm[::-1,index,:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
plt.colorbar(J_xz, ax=axes[1])
J_xy = axes[1].imshow(np.ones_like(J_norm[::-1,index,:]), alpha=0.8*Mask[::-1,index,:], cmap="Paired")
axes[1].set_title(fr"$||J|| \in ({index},\hat x, \hat z)$")
axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)



plt.savefig("guess.svg")
