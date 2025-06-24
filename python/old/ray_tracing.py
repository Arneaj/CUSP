import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, filename



B,V,J = import_from(filename)


J_norm = np.linalg.norm( J, axis=3 )
J_norm = J_norm[:,J_norm.shape[1]//2,:]


def interpolate(x, y, Bi):
    xm = int(x//1); ym = int(y//1)
    xd = x%1; yd = y%1

    B2d = Bi[xm:xm+2, ym:ym+2]
    B1d = B2d[0]*(1-xd) + B2d[1]*xd
    B0d = B1d[0]*(1-yd) + B1d[1]*yd

    return B0d


earth_pos = np.unravel_index( np.argmax(np.linalg.norm(B, axis=3)), B.shape[:-1] )

x = earth_pos[0]
dx = 1

while x>0 and x<B.shape[0]:
    if J_norm[x, J.shape[2]//2] > 1e-18:
        break
    x += dx

r_0 = 1.5*(x - earth_pos[0])    # TODO: FIX MAGIC NUMBER
alpha_0 = 0.6                   # TODO: FIX MAGIC NUMBER
# alpha_1 = 0.5
# alpha_2 = 0.5


Theta = np.linspace(-np.pi, np.pi, 200)
# Phi = np.linspace(0, np.pi, 100)
R = r_0 * ( 2 / (1 + np.cos(Theta)) )**(alpha_0)# + alpha_1*np.cos(Phi) + alpha_2*np.cos(Phi)**2)

X = earth_pos[0] - R * np.cos( Theta )     # np.sin( Phi )
Y = earth_pos[2] + R * np.sin( Theta )     # np.sin( Phi )
# Z = R * np.cos( Phi )


X_mask = (X>=0)
X = X[X_mask]; Y = Y[X_mask]#; Z = Z[X_mask]
R = R[X_mask]; Theta = Theta[X_mask]
X_mask = (X<B.shape[0])
X = X[X_mask]; Y = Y[X_mask]#; Z = Z[X_mask]
R = R[X_mask]; Theta = Theta[X_mask]

Y_mask = (Y>=0)
X = X[Y_mask]; Y = Y[Y_mask]#; Z = Z[Y_mask]
R = R[Y_mask]; Theta = Theta[Y_mask]
Y_mask = (Y<B.shape[2])
X = X[Y_mask]; Y = Y[Y_mask]#; Z = Z[Y_mask]
R = R[Y_mask]; Theta = Theta[Y_mask]





interest_points_x = []
interest_points_y = []

dr = 0.1

for i, theta in enumerate( Theta ):
    max_value = 0
    max_r = R[i]
    
    r = max_r
    x = earth_pos[0] - r*np.cos(theta)
    y = earth_pos[2] + r*np.sin(theta)
    
    while (x>=0) and (y>=0) and (x<B.shape[0]-1) and (y<B.shape[2]-1):
        # value = J_norm[int(x), int(y)]  # should be interpolate here but lazy rn
        value = interpolate( x, y, J_norm )
        
        if value > max_value:
            max_value = value
            max_r = r
            
        r += dr
        x = earth_pos[0] - r*np.cos(theta)
        y = earth_pos[2] + r*np.sin(theta)
        
    interest_points_x.append( earth_pos[0] - max_r*np.cos(theta) )
    interest_points_y.append( earth_pos[2] + max_r*np.sin(theta) )





plt.imshow( J_norm, cmap="inferno", vmin=0, vmax=5e-9 )
plt.plot(Y, X)
plt.plot(interest_points_y, interest_points_x)

plt.show()

