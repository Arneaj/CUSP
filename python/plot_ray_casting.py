import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from gorgon import import_from, filename
from earth_pos_detection import get_earth_pos

interest_points_x = []
interest_points_y = []
interest_points_z = []
interest_points_w = []

filepath = input("Enter the file path: ")


B = import_from(f"{filepath}/B_processed_real.txt")
B_norm = np.linalg.norm( B, axis=3 )

J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")

earth_pos = get_earth_pos( B_norm )


with open(f"{filepath}/interest_points_cpp.txt", "r") as f:
    lines = f.readlines()
    
    for line in lines:
        point = np.array( line.split(","), dtype=np.float32 )
        interest_points_x.append( earth_pos[0] - point[2] * np.cos(point[0]) )
        interest_points_y.append( earth_pos[1] + point[2] * np.sin(point[0]) * np.sin(point[1]) )
        interest_points_z.append( earth_pos[2] + point[2] * np.sin(point[0]) * np.cos(point[1]) )
        interest_points_w.append( (1-point[3], point[3], point[3]) )
        
    interest_points_x = np.array(interest_points_x)
    interest_points_y = np.array(interest_points_y)
    interest_points_z = np.array(interest_points_z)
    interest_points_w = np.array(interest_points_w)



def scatter_in_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', facecolor="black")

    ax.set_xlim(0, J_norm.shape[0])
    ax.set_ylim(0, J_norm.shape[1])
    ax.set_zlim(0, J_norm.shape[2])
    ax.set_axis_off()

    ax.scatter( interest_points_x, interest_points_y, interest_points_z, c=interest_points_w, alpha=0.1 )

    plt.show()


def get_animation(): 
    length = J_norm.shape[1]

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)


    J_norm_xy_i = J_norm[:,:,0]
    J_norm_xz_i = J_norm[:,0,:]

    epsilon = 1

    xy_points_x = interest_points_x[ np.abs(interest_points_z - 0) < epsilon ]
    xy_points_y = interest_points_y[ np.abs(interest_points_z - 0) < epsilon ]
    xy_c = interest_points_w[ np.abs(interest_points_z - 0) < epsilon ]

    xz_points_x = interest_points_x[ np.abs(interest_points_y - 0) < epsilon ]
    xz_points_z = interest_points_z[ np.abs(interest_points_y - 0) < epsilon ]
    xz_c = interest_points_w[ np.abs(interest_points_y - 0) < epsilon ]


    ### vmax
    # B: 1e-7
    # J: 5e-10
    # V: None


    J_xy = axes[0].imshow(J_norm_xy_i, cmap="inferno", vmin=0, vmax=5e-10)
    plt.colorbar(J_xy, ax=axes[0])
    J_xy = axes[0].scatter( xy_points_y, xy_points_x, s=3, c=xy_c )
    axes[0].set_title(fr"$||J||$ in $({0},\hat x, \hat y)$ plane")

    J_xz = axes[1].imshow(J_norm_xz_i, cmap="inferno", vmin=0, vmax=5e-9)
    plt.colorbar(J_xz, ax=axes[1])
    J_xz = axes[1].scatter( xz_points_z, xz_points_x, s=3, c=xz_c )
    axes[1].set_title(fr"$||J||$ in $({0},\hat z, \hat x)$ plane")


    axes[0].set_xlim(0, J_norm.shape[1])
    axes[0].set_ylim(0, J_norm.shape[0])
    axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")
    
    axes[1].set_xlim(0, J_norm.shape[2])
    axes[1].set_ylim(0, J_norm.shape[0])
    axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")



    def animate(i):        
        J_norm_xy_i = J_norm[:,:,int(i)]
        J_norm_xz_i = J_norm[:,int(i),:]
        
        xy_points_x = interest_points_x[ np.abs(interest_points_z - int(i)) < epsilon ]
        xy_points_y = interest_points_y[ np.abs(interest_points_z - int(i)) < epsilon ]
        xy_c = interest_points_w[ np.abs(interest_points_z - int(i)) < epsilon ]

        xz_points_x = interest_points_x[ np.abs(interest_points_y - int(i)) < epsilon ]
        xz_points_z = interest_points_z[ np.abs(interest_points_y - int(i)) < epsilon ]
        xz_c = interest_points_w[ np.abs(interest_points_y - int(i)) < epsilon ]

        axes[0].clear()
        J_xy = axes[0].imshow(J_norm_xy_i, cmap="inferno", vmin=0, vmax=5e-10)
        J_xy = axes[0].scatter( xy_points_y, xy_points_x, s=3, c=xy_c )
        axes[0].set_title(fr"$||J||$ in $({i},\hat x, \hat y)$ plane")
        axes[0].set_xlim(0, J_norm.shape[1])
        axes[0].set_ylim(0, J_norm.shape[0])

        axes[1].clear()
        J_xz = axes[1].imshow(J_norm_xz_i, cmap="inferno", vmin=0, vmax=5e-9)
        J_xz = axes[1].scatter( xz_points_z, xz_points_x, s=3, c=xz_c )
        axes[1].set_title(fr"$||J||$ in $({i},\hat x, \hat z)$ plane")
        axes[1].set_xlim(0, J_norm.shape[2])
        axes[1].set_ylim(0, J_norm.shape[0])
        
        return J_xy, J_xz
        
    ani = anim.FuncAnimation(fig, animate, interval=100, frames=np.arange(0, length, 2))  
    ani.save(filename=f"prediction_vs_reality_{filepath[-10:]}.gif", writer="pillow")

