import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from gorgon import filename, import_from, grad_mag_angle, gaussian_kernel_2d


if __name__ == "__main__":
    B,V,T,rho = import_from(filename)
    
    XMAX = 240
    XMIN = 0
    
    YMAX = 160
    YMIN = 0
    
    ZMAX = 160
    ZMIN = 0
    
    x = np.arange(0, XMAX-XMIN)
    y = np.arange(0, YMAX-YMIN)
    z = np.arange(0, ZMAX-ZMIN)
    
    
    V = V[XMIN:XMAX, YMIN:YMAX, 80]
    B = B[XMIN:XMAX, YMIN:YMAX, 80]
    # V = V[XMIN:XMAX, 80, ZMIN:ZMAX]
    # B = B[XMIN:XMAX, 80, ZMIN:ZMAX]
    
    
    bb = np.linalg.norm( B, axis = -1 )
    grad_bx_mag, _ = grad_mag_angle(B[:,:,0])
    
    
    LIMIT = 3e-8
    KERNEL_RAD = 5
    KERNEL_STD = 2.1
    gaussian_kern = gaussian_kernel_2d(KERNEL_RAD, KERNEL_STD)
    mask = signal.convolve2d( grad_bx_mag, gaussian_kern, "same" ) > LIMIT
    
    grad_mask_mag, _ = grad_mag_angle(mask)    
    norm_grad_mask_mag = grad_mask_mag/np.max(grad_mask_mag)
    
    
    plt.subplot(1, 3, 1)
    plt.title(r"Gradient magnitude")
    plt.imshow(grad_bx_mag, cmap="inferno", vmax=1e-7)
    # plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title(fr"Gradient magnitude $> {LIMIT}$")
    plt.imshow(mask, cmap="inferno")
    
    plt.subplot(1, 3, 3)
    plt.title(fr"Edge detected")
    plt.imshow(grad_bx_mag, cmap="inferno", vmax=1e-7)
    plt.imshow(np.ones_like(norm_grad_mask_mag), cmap="Greys", alpha=norm_grad_mask_mag)
    
    
    plt.show()
    






