import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from gorgon import import_from, get_gradients, signal


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)
    
if len(sys.argv) < 3:
    print("No vector name given!")
    exit(1)

filepath = sys.argv[1]
data_type = sys.argv[2]

vec = import_from(f"{filepath}/{data_type}_processed_real.txt")

kerning = 0

vec = vec[kerning:-1-kerning, kerning:-1-kerning, kerning:-1-kerning]


####### FIRST ORDER CRITICAL POINTS



x_is_positive = vec[:,:,:,0] >= 0
y_is_positive = vec[:,:,:,1] >= 0
z_is_positive = vec[:,:,:,2] >= 0

x_sign_change = signal.convolve(x_is_positive, np.ones((3,3,3)), "valid")
x_sign_change = x_sign_change * (27 - x_sign_change)

y_sign_change = signal.convolve(y_is_positive, np.ones((3,3,3)), "valid")
y_sign_change = y_sign_change * (27 - y_sign_change)

z_sign_change = signal.convolve(z_is_positive, np.ones((3,3,3)), "valid")
z_sign_change = z_sign_change * (27 - z_sign_change)


sign_change = x_sign_change * y_sign_change * z_sign_change
sign_change = ( sign_change - np.min(sign_change) ) / ( np.max(sign_change) - np.min(sign_change) )


##### JACOBIAN




critical_points = np.nonzero( sign_change > 0.01 )   

print(critical_points[0].shape)


dvecx_dx, dvecx_dy, dvecx_dz = get_gradients( vec[:,:,:,0], mode="valid" )
dvecy_dx, dvecy_dy, dvecy_dz = get_gradients( vec[:,:,:,1], mode="valid" )
dvecz_dx, dvecz_dy, dvecz_dz = get_gradients( vec[:,:,:,2], mode="valid" )


jacobian = np.zeros( ( critical_points[0].size, 3, 3 ) )
jacobian[:,0,0] = dvecx_dx[critical_points]
jacobian[:,0,1] = dvecx_dy[critical_points]
jacobian[:,0,2] = dvecx_dz[critical_points]

jacobian[:,1,0] = dvecy_dx[critical_points]
jacobian[:,1,1] = dvecy_dy[critical_points]
jacobian[:,1,2] = dvecy_dz[critical_points]

jacobian[:,2,0] = dvecz_dx[critical_points]
jacobian[:,2,1] = dvecz_dy[critical_points]
jacobian[:,2,2] = dvecz_dz[critical_points]

eigenvalues, eigenvectors = np.linalg.eig( jacobian )

sorted_indices = np.argsort( np.real(eigenvalues), axis=1 )

sorted_eigenvalues = [ eigenvalues[i][sorted_indices[i]] for i in range(eigenvalues.shape[0]) ]
sorted_eigenvectors = [ eigenvectors[i][sorted_indices[i]] for i in range(eigenvalues.shape[0]) ]

eigenvalues = np.array(sorted_eigenvalues)
eigenvectors = np.array(sorted_eigenvectors)


classes = { 0:"source", 1:"repelling saddle", 2:"attracting saddle", 3:"sink" }
subclasses = { 0:"foci", 1:"nodes" }

classification = (
    0 * (np.real(eigenvalues[:,0])>0) + 
    1 * (np.real(eigenvalues[:,0])<0)*(np.real(eigenvalues[:,1])>0) +
    2 * (np.real(eigenvalues[:,1])<0)*(np.real(eigenvalues[:,2])>0) +
    3 * (np.real(eigenvalues[:,2])<0)
)

node_indices = (np.imag(eigenvalues[:,0])==0)*(np.imag(eigenvalues[:,1])==0)*(np.imag(eigenvalues[:,2])==0)

classification += (
    0 * 4 * (1-node_indices) + 
    1 * 4 * node_indices
)


critical_points_x = critical_points[0]
critical_points_y = critical_points[1]
critical_points_z = critical_points[2]

fig = plt.figure(); ax = fig.add_subplot(projection="3d")
fig.set_figwidth(6)
fig.set_figheight(6)

ax.set_xlim( 0, vec.shape[0]-1 )
ax.set_ylim( 0, vec.shape[1]-1 )
ax.set_zlim( 0, vec.shape[2]-1 )

ax.set_xlabel("$x$ [$R_E$]")
ax.set_ylabel("$y$ [$R_E$]")
ax.set_zlabel("$z$ [$R_E$]")

ax.view_init(elev=0., azim=270)

    
for j in range(4, 8):
    p_x = critical_points_x[classification == j]
    p_y = critical_points_y[classification == j]
    p_z = critical_points_z[classification == j]
    
    ax.scatter( p_x, p_y, p_z, color=(1-j/7, j//4, j/7), label=f"{classes[j%4]} {subclasses[j//4]}" )

    
fig.legend()

plt.savefig("../images/skeleton_classification.png")


