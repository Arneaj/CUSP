import numpy as np
import matplotlib.pyplot as plt

from gorgon import import_from, filename

import sys


#last line deletion
def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')



B,V,J = import_from(filename)


J_norm = np.linalg.norm( J, axis=3 )


def interpolate(x, y, z, Bi):
    xm = int(x//1); ym = int(y//1); zm = int(z//1)
    xd = x%1; yd = y%1; zd = z%1

    B3d = Bi[xm:xm+2, ym:ym+2, zm:zm+2]
    B2d = B3d[0]*(1-xd) + B3d[1]*xd
    B1d = B2d[0]*(1-yd) + B2d[1]*yd
    B0d = B1d[0]*(1-zd) + B1d[1]*zd

    return B0d


earth_pos = np.unravel_index( np.argmax(np.linalg.norm(B, axis=3)), B.shape[:-1] )

x = earth_pos[0]
dx = 1



while x>0 and x<B.shape[0]:
    if J_norm[x, J.shape[1]//2, J.shape[2]//2] > 1e-18:
        break
    x += dx




r_0 = 1.5*(x - earth_pos[0])    # TODO: FIX MAGIC NUMBER
alpha_0 = 0.6                   # TODO: FIX MAGIC NUMBER
alpha_1 = 0
alpha_2 = 0


Theta = np.linspace(-np.pi, np.pi, 100)
Phi = np.linspace(0, np.pi, 100)
R = r_0 * ( 2 / (1 + np.cos(Theta)) )**(alpha_0)# + alpha_1*np.cos(Phi) + alpha_2*np.cos(Phi)**2)


R_mask = (R != np.nan)
R = R[R_mask]
Theta = Theta[R_mask]

R_mask = (R != np.inf)
R = R[R_mask]
Theta = Theta[R_mask]




interest_points_x = []
interest_points_y = []
interest_points_z= []

dr = 0.1

for i, theta in enumerate( Theta ):
    for j, phi in enumerate( Phi ):
        max_value = 0
        max_r = R[i]
        
        r = max_r
        x = earth_pos[0] - r*np.cos(theta)   #:*np.sin(phi)
        y = earth_pos[1] + r*np.sin(theta)*np.sin(phi)
        z = earth_pos[2] + r*np.sin(theta)*np.cos(phi)
        
        while (x>=0) and (y>=0) and (z>=0) and (x<B.shape[0]-1) and (y<B.shape[1]-1) and (z<B.shape[2]-1):
            value = interpolate( x, y, z, J_norm )
            
            if value > max_value:
                max_value = value
                max_r = r
                
            r += dr
            x = earth_pos[0] - r*np.cos(theta)   #:*np.sin(phi)
            y = earth_pos[1] + r*np.sin(theta)*np.sin(phi)
            z = earth_pos[2] + r*np.sin(theta)*np.cos(phi)
            
        if max_r == R[i]: continue
            
        interest_points_x.append( earth_pos[0] - max_r*np.cos(theta) )
        interest_points_y.append( earth_pos[1] + max_r*np.sin(theta)*np.sin(phi) )
        interest_points_z.append( earth_pos[2] + max_r*np.sin(theta)*np.cos(phi) )
    
    if (i>0): delete_last_line()
    print(f"{(100*(i+1)/len(Theta)):.2f}% done.")




# plt.imshow( J_norm, cmap="inferno", vmin=0, vmax=5e-9 )
# plt.plot(Y, X)
# plt.plot(interest_points_y, interest_points_x)


fig = plt.figure()
ax = fig.add_subplot(projection='3d', facecolor="black")

ax.set_xlim(0, J_norm.shape[0])
ax.set_ylim(0, J_norm.shape[1])
ax.set_zlim(0, J_norm.shape[2])
ax.set_axis_off()


with open("../data/interest_points.txt", "w") as f:
    f.write( str( interest_points_x[0] ) )
    for i in range(1, len(interest_points_x)):
        f.write(",")
        f.write( str( interest_points_x[i] ) )
    f.write("\n")
    
    f.write( str( interest_points_y[0] ) )
    for i in range(1, len(interest_points_y)):
        f.write(",")
        f.write( str( interest_points_y[i] ) )
    f.write("\n") 
    
    f.write( str( interest_points_z[0] ) )
    for i in range(1, len(interest_points_z)):
        f.write(",")
        f.write( str( interest_points_z[i] ) )
    f.write("\n")


# ax.scatter( X, Y, Z )

ax.scatter( interest_points_x, interest_points_y, interest_points_z, c="white", alpha=0.2 )


plt.show()

