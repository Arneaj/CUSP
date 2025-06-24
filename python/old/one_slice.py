import h5py

import matplotlib.pyplot as plt

import numpy as np

filename = "../data/Gorgon_20210420__MS_params_7580s.hdf5"


def one_slice():
    f = h5py.File(filename, "r")

    keys = list(f.keys())
    B_key = keys[0]
    V_key = keys[3]
    T_key = keys[1]
    rho_key = keys[2]

    B = f[B_key][()]
    V = f[V_key][()]
    T = f[T_key][()] 
    rho = f[rho_key][()]
    
    f.close()
        
    # XMAX = 126
    # XMIN = 30
    
    # YMAX = 135
    # YMIN = 25
    
    # ZMAX = 106
    # ZMIN = 54
    
    XMAX = 240
    XMIN = 0

    YMAX = 160
    YMIN = 0

    ZMAX = 160
    ZMIN = 0
    
    x = np.arange(0, XMAX-XMIN)
    y = np.arange(0, YMAX-YMIN)
    z = np.arange(0, ZMAX-ZMIN)
    
    length = B.shape[1]
    
    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)
    
    B_t_norm = np.transpose(np.linalg.norm(B[XMIN:XMAX,YMIN:YMAX,ZMIN:ZMAX], axis=3), axes=(2,0,1))
    B_t_x = np.transpose(B[XMIN:XMAX,YMIN:YMAX,ZMIN:ZMAX,0], axes=(2,0,1))
    B_t_y = np.transpose(B[XMIN:XMAX,YMIN:YMAX,ZMIN:ZMAX,1], axes=(2,0,1))
    B_t_z = np.transpose(B[XMIN:XMAX,YMIN:YMAX,ZMIN:ZMAX,2], axes=(2,0,1))
    
    layer_z = (ZMAX-ZMIN)//2
    layer_y = (YMAX-YMIN)//2
    
    B_t_norm_xy_i = B_t_norm[layer_z,:,:]
    B_t_norm_zx_i = B_t_norm[:,:,layer_y]
    

    levels = 10**( np.linspace( np.log10(np.min(B_t_norm)) , -7, 10) )
    
    
    Bz_xy = axes[0].imshow(B_t_norm_xy_i, cmap="copper", vmin=0, vmax=1e-7)
    plt.colorbar(Bz_xy, ax=axes[0])
    axes[0].set_title(r"$||B||$ in $(i,\hat{y}, \hat{x})$ plane")
    #axes[0].contour(B_t_norm_xy_i, colors=["white"], levels=[5e-10])
    # axes[0].streamplot(y, x, B_t_y[layer_z,:,:], B_t_x[layer_z,:,:],
    #                     density=1, color=(0.8,0.8,0.8),
    #                     minlength=0.1, 
    #                     arrowstyle="-",
    #                     broken_streamlines=False
    #                     )
    axes[0].contour(B_t_norm_xy_i, vmin=0, vmax=1e-7, levels=levels)
    
    
    

    Bz_zx = axes[1].imshow(np.transpose(B_t_norm_zx_i), cmap="copper", vmin=0, vmax=1e-7)
    plt.colorbar(Bz_zx, ax=axes[1])
    axes[1].set_title(r"$||B||$ in $(i,\hat{z}, \hat{x})$ plane")
    # axes[1].streamplot(z, x, np.transpose(B_t_z[:,:,layer_y]), np.transpose(B_t_x[:,:,layer_y]),
    #                     density=2, color=(0.8,0.8,0.8),
    #                     minlength=0.05, 
    #                     arrowstyle="-",
    #                     broken_streamlines=False
    #                     )
    axes[1].contour(np.transpose(B_t_norm_zx_i), vmin=0, vmax=1e-7, levels=levels)

    # axes[1].scatter([79.1, 79], [48.4, 87.7], marker="x", color="red")
    # axes[1].plot([79.1, 79, 77.9, 75.5, 79.1, 74.7, 75.5, 69.1, 69.5],
    #              [48.4, 87.7, 121.1, 169.9, 174.4, 181.9, 191.5, 225.3, 233.3],
    #              color="red")
    
    # 79.1, 48.4 dayside reconnection
    # 79, 87.7 nightside reconnection
    
    # 77.9, 121.1 
    # 75.5, 169.6
    # 79.1, 174.4
    # 74.7, 181.9
    # 75.5, 191.5
    # 69.1, 225.3
    # 69.5, 233.3
    
    axes[0].set_xlim(0,YMAX-YMIN-1)
    axes[0].set_ylim(0,XMAX-XMIN-1)
    axes[0].set_xlabel(r"$y / R_E$")
    axes[0].set_ylabel(r"$x / R_E$")
    
    axes[1].set_xlim(0,ZMAX-ZMIN-1)
    axes[1].set_ylim(0,XMAX-XMIN-1)
    axes[1].set_xlabel(r"$z / R_E$")
    axes[1].set_ylabel(r"$x / R_E$")
    
    
    plt.show()



if __name__ == "__main__":
    one_slice()