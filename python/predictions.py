import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from gorgon import import_from, spherical_to_cartesian, import_from_bin
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

# J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
J_norm = import_from_bin(f"{filepath}/J_processed_real.bin")

J_norm = np.linalg.norm( J_norm, axis=3 )


# earth_pos = get_earth_pos( B_norm )
# earth_pos = [29.75, 58, 58]
earth_pos = [30.75, 58, 58]

interest_points_theta = []
interest_points_phi = []
interest_points_r = []
interest_points_w = []

with open(f"{filepath}/interest_points_cpp.txt", "r") as f:
    lines = f.readlines()

    for line in lines:
        point = np.array( line.split(","), dtype=np.float32 )
        interest_points_theta.append( point[0] )
        interest_points_phi.append( point[1] )
        interest_points_r.append( point[2] )
        interest_points_w.append( (1-point[3], point[3], point[3]) )

    interest_points_theta = np.array(interest_points_theta)
    interest_points_phi = np.array(interest_points_phi)
    interest_points_r = np.array(interest_points_r)
    interest_points_w = np.array(interest_points_w)


nb_phi = 0



# B = import_from(f"{filepath}/B_processed_sim.txt")
# B_norm = np.linalg.norm(B, axis=3)

# X = import_from(f"{filepath}/X.txt")
# Y = import_from(f"{filepath}/Y.txt")
# Z = import_from(f"{filepath}/Z.txt")

# shape = B_norm.shape
# earth_pos_tilde = get_earth_pos(B_norm)

# x_min = np.array([np.min(X), np.min(Y), np.min(Z)])
# x_max = np.array([np.max(X), np.max(Y), np.max(Z)])
# dx = x_max - x_min


# x_tilde = earth_pos_tilde[0] + interest_points_r * np.cos(interest_points_theta)
# y_tilde = earth_pos_tilde[1] + interest_points_r * np.sin(interest_points_theta) * np.sin(interest_points_phi)
# z_tilde = earth_pos_tilde[2] + interest_points_r * np.sin(interest_points_theta) * np.cos(interest_points_phi)

# X = x_tilde * dx[0] / shape[0] + x_min[0]
# Y = y_tilde * dx[1] / shape[1] + x_min[1]
# Z = z_tilde * dx[2] / shape[2] + x_min[2]

# X += J_norm.shape[0] - earth_pos[0]
# Y += J_norm.shape[1] - earth_pos[1]
# Z += J_norm.shape[2] - earth_pos[2]

X = J_norm.shape[0] - earth_pos[0] + interest_points_r * np.cos(interest_points_theta)
Y = earth_pos[1] + interest_points_r * np.sin(interest_points_theta) * np.sin(interest_points_phi)
Z = earth_pos[2] + interest_points_r * np.sin(interest_points_theta) * np.cos(interest_points_phi)

# length = J_norm.shape[1]

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(10)
fig.set_figheight(5.5)

iy = 58

J_norm_xy_i = J_norm[::-1,:,iy]
J_norm_xz_i = J_norm[::-1,iy,:]

epsilon = 1

xy_points_x = X[ np.abs(Z - iy) < epsilon ]
xy_points_y = Y[ np.abs(Z - iy) < epsilon ]
xy_c = interest_points_w[ np.abs(Z - iy) < epsilon ]

xz_points_x = X[ np.abs(Y - iy) < epsilon ]
xz_points_z = Z[ np.abs(Y - iy) < epsilon ]
xz_c = interest_points_w[ np.abs(Y - iy) < epsilon ]


### vmax
# B: 1e-7
# J: 5e-10
# V: None


J_xy = axes[0].imshow(J_norm_xy_i, cmap="inferno", vmin=0, vmax=3e-9, interpolation="none")
plt.colorbar(J_xy, ax=axes[0])
J_xy = axes[0].scatter( xy_points_y, xy_points_x, s=0.3, c=xy_c )
axes[0].set_title(fr"$||J||$ in $({58},\hat x, \hat y)$ plane")

J_xz = axes[1].imshow(J_norm_xz_i, cmap="inferno", vmin=0, vmax=3e-9, interpolation="none")
plt.colorbar(J_xz, ax=axes[1])
J_xz = axes[1].scatter( xz_points_z, xz_points_x, s=0.3, c=xz_c )
axes[1].set_title(fr"$||J||$ in $({58},\hat z, \hat x)$ plane")


axes[0].set_xlim(0, J_norm.shape[1]-1)
axes[0].set_ylim(0, J_norm.shape[0]-1)
axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")


axes[1].set_xlim(0, J_norm.shape[2]-1)
axes[1].set_ylim(0, J_norm.shape[0]-1)
axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")


name_of_datapoint = filepath.split("/")[-1]

run_nb, timestep = name_of_datapoint.split("_")


fig.suptitle(f"{run_nb} at timestep $t =$ {timestep}s")



plt.savefig("../images/predictions.svg")


