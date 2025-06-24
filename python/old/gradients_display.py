import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from gorgon import filename, import_from, grad_mag_angle


if __name__ == "__main__":
    B,V,T,rho = import_from(filename)
    
    XMAX = 240
    XMIN = 0
    
    YMAX = 160
    YMIN = 0
    
    ZMAX = 160
    ZMIN = 0
    
    # XMAX = 126
    # XMIN = 30
    
    # YMAX = 135
    # YMIN = 25
    
    # ZMAX = 106
    # ZMIN = 54
    
    x = np.arange(0, XMAX-XMIN)
    y = np.arange(0, YMAX-YMIN)
    z = np.arange(0, ZMAX-ZMIN)
    
    
    V = V[XMIN:XMAX, YMIN:YMAX, 80]
    B = B[XMIN:XMAX, YMIN:YMAX, 80]
    # V = V[XMIN:XMAX, 80, ZMIN:ZMAX]
    # B = B[XMIN:XMAX, 80, ZMIN:ZMAX]
    
    
    vv = np.linalg.norm( V, axis = -1 )
    bb = np.linalg.norm( B, axis = -1 )
    
    
    scharr = np.array([ 
                        [ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]
                      ]) # Gx + j*Gy
    
    grad_bz_mag, grad_bz_angle = grad_mag_angle(B[:,:,2])
    grad_bx_mag, grad_bx_angle = grad_mag_angle(B[:,:,0])



    plt.subplot(2, 3, 1)
    plt.title(r"$B_z$")
    plt.imshow(B[:,:,2], cmap="inferno", vmax=1e-7, vmin=-1e-7)
    plt.colorbar()
     
    
    plt.subplot(2, 3, 2)
    plt.title(r"Gradient magnitude")
    plt.imshow(grad_bz_mag, cmap="inferno", vmax=1e-7)
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.title(r"Gradient orientation")
    plt.imshow(grad_bz_angle, cmap="twilight")
    plt.colorbar()
    
    
    
    
    
    plt.subplot(2, 3, 4)
    plt.title(r"$B_x$")
    plt.imshow(B[:,:,0], cmap="inferno", vmax=1e-7, vmin=-1e-7)
    plt.colorbar()
     
    
    plt.subplot(2, 3, 5)
    plt.title(r"Gradient magnitude")
    plt.imshow(grad_bx_mag, cmap="inferno", vmax=1e-7)
    plt.colorbar()
    
    plt.subplot(2, 3, 6)
    plt.title(r"Gradient orientation")
    plt.imshow(grad_bx_angle, cmap="twilight")
    plt.colorbar()
    
    
    
    
    
    
    
    # plt.streamplot(y, x, V[XMIN:XMAX, YMIN:YMAX, 80, 1], V[XMIN:XMAX, YMIN:YMAX, 80, 0],
    #                     density=1.5, color=(0.8,0.8,0.8),
    #                     minlength=0.1, 
    #                     arrowstyle="-",
    #                     broken_streamlines=False
    #                     )
    
    # plt.streamplot(y, x, V[XMIN:XMAX, YMIN:YMAX, 80, 1], V[XMIN:XMAX, YMIN:YMAX, 80, 0],
    #                     density=1.5, color=(0.8,0.8,0.8),
    #                     minlength=0.1, 
    #                     arrowstyle="-",
    #                     broken_streamlines=False
    #                     )
    
    plt.show()
    






