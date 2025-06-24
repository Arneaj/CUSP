
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import cm

import numpy as np

from gorgon import import_from, filename



def slices_2D():
    _,_,B = import_from(filename)
    
    # x = np.arange(240)
    # y = np.arange(160)
    # z = np.arange(160)
    
    length = B.shape[1]
    
    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)
    
    # B_t_norm = np.transpose(B, axes=(2,0,1))
    
    B_t_norm = np.linalg.norm(B, axis=3)
    # B_t_x = np.transpose(B[:,:,:,0], axes=(2,0,1))
    # B_t_y = np.transpose(B[:,:,:,1], axes=(2,0,1))
    # B_t_z = np.transpose(B[:,:,:,2], axes=(2,0,1))
    
    B_t_norm_xy_i = B_t_norm[:,:,0]
    B_t_norm_zx_i = B_t_norm[:,0,:]
    
    
    ### vmax
    # B: 1e-7
    # J: 5e-10
    # V: None
    
    
    Bz_xy = axes[0].imshow(B_t_norm_xy_i, cmap="inferno", vmin=0, vmax=5e-9)
    plt.colorbar(Bz_xy, ax=axes[0])
    axes[0].set_title(fr"$||J||$ in $({0},\hat x, \hat y)$ plane")
    # axes[0].streamplot(y, x, B_t_x[0,:,:], B_t_y[0,:,:], density=0.4, color="white")
    #axes[0].contour(B_t_norm_xy_i, colors=["white"], levels=[2e-10])

    Bz_zx = axes[1].imshow(B_t_norm_zx_i, cmap="inferno", vmin=0, vmax=5e-9)
    plt.colorbar(Bz_zx, ax=axes[1])
    axes[1].set_title(fr"$||J||$ in $({0},\hat z, \hat x)$ plane")
    # axes[1].streamplot(x, z, B_t_z[:,:,0], B_t_x[:,:,0], density=0.4, color="white")
    
    
    def animate(i):        
        B_t_norm_xy_i = B_t_norm[:,:,int(i)]
        B_t_norm_zx_i = B_t_norm[:,int(i),:]

        axes[0].clear()
        Bz_xy = axes[0].imshow(B_t_norm_xy_i, cmap="inferno", vmin=0, vmax=5e-9)
        axes[0].set_title(fr"$||J||$ in $({i},\hat x, \hat y)$ plane")
        # axes[0].streamplot(y, x, B_t_x[i,:,:], B_t_y[i,:,:], density=0.4, color="white")
        #axes[0].contour(B_t_norm_xy_i, colors=["white"], levels=[2e-10])

        axes[1].clear()
        Bz_zx = axes[1].imshow(B_t_norm_zx_i, cmap="inferno", vmin=0, vmax=5e-9)
        axes[1].set_title(fr"$||J||$ in $({i},\hat x, \hat z)$ plane")
        # axes[1].streamplot(x, z, B_t_z[:,:,i], B_t_x[:,:,i], density=0.4, color="white")
        
        return Bz_xy, Bz_zx
        
    ani = anim.FuncAnimation(fig, animate, interval=150, frames=np.arange(0, length))  
    ani.save(filename="new_J_processed_1p2.gif", writer="pillow")





if __name__ == "__main__":  
    slices_2D()
