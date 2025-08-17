import spacepy.omni as om
from spacepy.pybats import IdlFile
from spacepy.pybats.bats import MagGridFile, Bats2d
from scipy.interpolate import griddata
import pandas as pd
import numpy as np

# spacepy.toolbox.update(QDomni=True)
 
# Load an arbitrary IDL-formatted file naively:
# raw = IdlFile('/rds/general/user/avr24/home/Thesis/SWMF/MAGNETOMETER_on_GRID%2Foutputs%2Fmag_grid_e20241009-180000.out')
 
# Use subclasses to get output-specific functionality:
mhd = Bats2d('/rds/general/user/avr24/home/Thesis/SWMF/get_run_file.php?runnumber=Gregory_Kennedy_072525_GM_4&filename=GM%2FIO2%2F3d__var_1_e20000101-090000-000.out')
# mag = MagGridFile('/rds/general/user/avr24/home/Thesis/SWMF/MAGNETOMETER_on_GRID%2Foutputs%2Fmag_grid_e20241009-180000.out', iframe=0)

# print(mag.attrs)

# Load matplotlib, create a figure:
# import matplotlib.pyplot as plt
# fig=plt.figure()

print( mhd.attrs )

print( mhd.keys() )

mhd.calc_j()

grid = mhd["grid"]
x = mhd["x"]
y = mhd["y"]
z = mhd["z"]
J = mhd["j"]
Rho = mhd["rho"]

# earth_pos = mhd.

J = np.array(J)
Rho = np.array(Rho)

x = np.array(x)
y = np.array(y)
z = np.array(z)

# Create a regular grid for interpolation
# Adjust these ranges and resolution as needed
reduce_factor = 2

x_min, x_max = x.min() // reduce_factor, x.max()
y_min, y_max = y.min() // (reduce_factor), y.max() // (reduce_factor)
z_min, z_max = z.min() // (reduce_factor), z.max() // (reduce_factor)

extra_precision = 1.0

# Create regular grid points
nx, ny, nz = int(extra_precision * (x_max-x_min)), int(extra_precision * (y_max-y_min)), int(extra_precision * (z_max-z_min))  # Adjust resolution as needed
xi = np.linspace(x_min, x_max, nx)
yi = np.linspace(y_min, y_max, ny) 
zi = np.linspace(z_min, z_max, nz)

# Create meshgrid
Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='ij')

# Prepare points for interpolation
points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
values_J = J.ravel()
values_Rho = Rho.ravel()

print("Finished reading files")

# Interpolate onto regular grid
J_regrid = griddata(points, values_J, (Xi, Yi, Zi), method='nearest')[::-1,:,:]
Rho_regrid = griddata(points, values_Rho, (Xi, Yi, Zi), method='nearest')[::-1,:,:]
earth_pos = extra_precision * np.array([ x_max, y_max, z_max ])
# earth_pos = extra_precision * np.array([ -x_min, -y_min, -z_min ])

print("Finished processing files")

print(f"Original data shape: {J.shape}")
print(f"Regridded data shape: {J_regrid.shape}")
print(f"Earth position: {earth_pos}")
print(f"J[earth_pos]: {J_regrid[int(earth_pos[0]), int(earth_pos[1]), int(earth_pos[2])]}")

import mag_cusps as cusps
import gorgon


MP = cusps.get_interest_points(
    J_regrid, earth_pos, 
    Rho_regrid,
    theta_min=0.0, theta_max=np.pi*0.85,  
    nb_theta=30, nb_phi=30,
    dx=0.1, dr=0.1,
    alpha_0_min=0.4, alpha_0_max=0.6, nb_alpha_0=4,
    r_0_mult_min=1.0, r_0_mult_max=2.5, nb_r_0=20
)

# MP_params, MP_cost = cusp.fit_to_Rolland25( 
#     MP, MP.shape[0],               # r_0                        a_0     a_1     a_2     d_n                     l_n     s_n     d_s                     l_s     s_s     e         
#     initial_params      = np.array([ extra_precision * 10.0,    0.5,    0,      0,      extra_precision * 3,    0.55,   5,      extra_precision * 3,    0.55,   5,      0 ]),
#     lowerbound          = np.array([ extra_precision * 5.0,     0.2,    -1.0,   -1.0,   extra_precision * 0,    0.1,    0.1,    extra_precision * 0,    0.1,    0.1,    -0.8 ]),
#     upperbound          = np.array([ extra_precision * 15.0,    0.8,    1.0,    1.0,    extra_precision * 6,    2,      10,     extra_precision * 6,    2,      10,     0.8 ]),
#     radii_of_variation  = np.array([ extra_precision * 3.0,     0.2,    0.5,    0.5,    extra_precision * 2,    0.1,    3,      extra_precision * 2,    0.1,    3,      0.5 ]),
# )




X_MP, Y_MP, Z_MP = gorgon.spherical_to_cartesian( MP[:,2], MP[:,0], MP[:,1], earth_pos )

is_in_plane_MP = np.abs(Y_MP-int(earth_pos[1])) < 1

X_MP_plot = X_MP[is_in_plane_MP]
Z_MP_plot = Z_MP[is_in_plane_MP]




# Theta = np.linspace(0, np.pi*0.99, 200)
# Phi = 0
# R1 = gorgon.Me25_poly( MP_params, Theta, Phi )
# X11, _, Z11 = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
# Phi = np.pi
# R1 = gorgon.Me25_poly( MP_params, Theta, Phi )
# X12, _, Z12 = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
# X1 = np.concatenate( [X12[::-1], X11] )
# Z1 = np.concatenate( [Z12[::-1], Z11] )

# X1_is_in = (X1 >= 0) * (X1 < J_regrid.shape[0])

# X1 = X1[X1_is_in]
# Z1 = Z1[X1_is_in]


import matplotlib.pyplot as plt


# plt.imshow( np.moveaxis( J_regrid[:,ny//2,:], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-3, interpolation="none" )
plt.imshow( np.moveaxis( Rho_regrid[:,ny//2,:], [0,1], [1,0] ), cmap="inferno", norm="log", interpolation="none" )
plt.colorbar()

# plt.scatter(earth_pos[0], earth_pos[2])

plt.scatter( X_MP_plot, Z_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
# plt.plot(X1, Z1)

plt.savefig("../images/swmf.svg")
