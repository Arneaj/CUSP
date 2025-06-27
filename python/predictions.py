import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from gorgon import import_from, filename
from earth_pos_detection import get_earth_pos


interest_points_x = []
interest_points_y = []
interest_points_z = []
interest_points_w = []


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]


# B = import_from(f"{filepath}/B_processed_real.txt")
# B_norm = np.linalg.norm( B, axis=3 )

J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")

# earth_pos = get_earth_pos( B_norm )
earth_pos = [33, 58, 58]

with open(f"{filepath}/interest_points_cpp.txt", "r") as f:
    lines = f.readlines()

    for line in lines:
        point = np.array( line.split(","), dtype=np.float32 )
        interest_points_x.append( J_norm.shape[0] - earth_pos[0] + point[2] * np.cos(point[0]) )
        interest_points_y.append( earth_pos[1] + point[2] * np.sin(point[0]) * np.sin(point[1]) )
        interest_points_z.append( earth_pos[2] + point[2] * np.sin(point[0]) * np.cos(point[1]) )
        interest_points_w.append( (1-point[3], point[3], point[3]) )

    interest_points_x = np.array(interest_points_x)
    interest_points_y = np.array(interest_points_y)
    interest_points_z = np.array(interest_points_z)
    interest_points_w = np.array(interest_points_w)


# length = J_norm.shape[1]

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(10)
fig.set_figheight(4)

iy = 58

J_norm_xy_i = J_norm[::-1,:,iy]
J_norm_xz_i = J_norm[::-1,iy,:]

epsilon = 1

xy_points_x = interest_points_x[ np.abs(interest_points_z - iy) < epsilon ]
xy_points_y = interest_points_y[ np.abs(interest_points_z - iy) < epsilon ]
xy_c = interest_points_w[ np.abs(interest_points_z - iy) < epsilon ]

xz_points_x = interest_points_x[ np.abs(interest_points_y - iy) < epsilon ]
xz_points_z = interest_points_z[ np.abs(interest_points_y - iy) < epsilon ]
xz_c = interest_points_w[ np.abs(interest_points_y - iy) < epsilon ]


### vmax
# B: 1e-7
# J: 5e-10
# V: None


J_xy = axes[0].imshow(J_norm_xy_i, cmap="inferno", vmin=0, vmax=3e-9, interpolation="none")
plt.colorbar(J_xy, ax=axes[0])
J_xy = axes[0].scatter( xy_points_y, xy_points_x, s=2, c=xy_c )
axes[0].set_title(fr"$||J||$ in $({58},\hat x, \hat y)$ plane")

J_xz = axes[1].imshow(J_norm_xz_i, cmap="inferno", vmin=0, vmax=3e-9, interpolation="none")
plt.colorbar(J_xz, ax=axes[1])
J_xz = axes[1].scatter( xz_points_z, xz_points_x, s=2, c=xz_c )
axes[1].set_title(fr"$||J||$ in $({58},\hat z, \hat x)$ plane")


axes[0].set_xlim(0, J_norm.shape[1]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)
axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")


axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)
axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")

plt.savefig("predictions.svg")


