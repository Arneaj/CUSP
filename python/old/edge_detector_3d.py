import h5py

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from scipy import signal

from gorgon import filename, import_from, grad_mag_angle, gaussian_kernel_2d, grad_mag_3d, gaussian_kernel_3d

from streamlines_3d import plot_smooth_streamlines_pos


if __name__ == "__main__":
    B,V,T,rho = import_from(filename)

    XMAX = 240
    XMIN = 0

    YMAX = 160
    YMIN = 0

    ZMAX = 160
    ZMIN = 0

    STEP = 2

    x = np.arange(0, XMAX-XMIN, STEP)
    y = np.arange(0, YMAX-YMIN, STEP)
    z = np.arange(0, ZMAX-ZMIN, STEP)


    AXIS = 0


    grad_3d = grad_mag_3d( B[::STEP,::STEP,::STEP, AXIS] )
    
    print("first grad done")
    
    
    LIMIT = 3e-8                                ### TODO: FIX MAGIC NUMBER
    KERNEL_RAD = 4#6                              ### TODO: FIX MAGIC NUMBER
    KERNEL_STD = 3.2 * np.sqrt(STEP) # 2.1 * np.sqrt(STEP)            ### TODO: FIX MAGIC NUMBER ?
    gaussian_kern = gaussian_kernel_3d(KERNEL_RAD, KERNEL_STD)
    
    mask_3d = signal.convolve( grad_3d, gaussian_kern, "same" ) > LIMIT
        
    print("mask done")
    
    
    grad_mask_3d = grad_mag_3d(mask_3d)
    
    max_val = np.max(grad_mask_3d)
    
    if max_val > 0:
        grad_mask_3d = grad_mask_3d / max_val
    
    print("second grad done")
    
    
    plot_mask = grad_mask_3d > 0.8               ### TODO: FIX MAGIC NUMBER ?
    
    Z,X,Y = np.meshgrid(z,x,y)
    X = X[plot_mask]
    Y = Y[plot_mask]
    Z = Z[plot_mask]
    
    print("second mask done")
    
    
    # ax = plt.figure().add_subplot(projection='3d', facecolor='black')
    
    # ax.set_xlim(0, B.shape[0])
    # ax.set_ylim(0, B.shape[1])
    # ax.set_zlim(0, B.shape[2])
    # ax.set_axis_off()
    
    # base_pos = np.array([ 60-XMIN, (YMAX-YMIN)/2, (ZMAX-ZMIN)/2 ])
    # positions = []
    # NB_POINTS = 4
    # RADIUS = 15
    # for ix in range(NB_POINTS):
    #     for iy in range(NB_POINTS):
    #         for iz in range(NB_POINTS):
    #             positions.append( base_pos + np.array([ix,iy,iz]) * 2*RADIUS/(NB_POINTS-1) - RADIUS )
    
    # plot_smooth_streamlines_pos(ax, 
    #                             B[:,:,:,0],
    #                             B[:,:,:,1],
    #                             B[:,:,:,2], 
    #                             positions, 1)
    
    # print("finished streamlines")
    
    # ax.scatter(X, Y, Z, color=(1,0,0, 0.1))
    
    # print("finished scattering")
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    
    frame = ax.imshow( B[::STEP,0,::STEP, AXIS], cmap="copper", vmax=1e-8 )
    ax.imshow( np.ones_like(B[::STEP,0,::STEP, AXIS]), cmap="Greys", alpha=np.array(plot_mask[:,0,:], dtype=np.float32) )
    
    def animate(i):
        fig.suptitle(f"y index = {i}")
        
        ax.clear()
        frame = ax.imshow( B[::STEP,i*STEP,::STEP, AXIS], cmap="copper", vmax=1e-8 )
        ax.imshow( np.ones_like(B[::STEP,i*STEP,::STEP, AXIS]), cmap="Greys", alpha=np.array(plot_mask[:,i,:], dtype=np.float32) )
        
        if i % 10 == 0:
            print(f"frame {i} done")
        
        return frame
        
    ani = anim.FuncAnimation(fig, animate, interval=50, frames=np.arange(0, 160//STEP))  
    ani.save(filename="thing.gif", writer="pillow")





