import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.cm as cm
import matplotlib as mpl

from gorgon import import_from, Me25, spherical_to_cartesian, get_gradients, interpolate, signal, Me25_leaky
from earth_pos_detection import get_earth_pos


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)


filepath = sys.argv[1]


B = import_from(f"{filepath}/B_processed_real.txt")
B_norm = np.linalg.norm( B, axis=3 )

J_norm = None
V = None

earth_pos = get_earth_pos( B_norm )
print( "earth pos = ", earth_pos )

shape = B_norm.shape
print( "shape = ", B_norm.shape )

# shape = (216, 120, 120)
# earth_pos = (41, 60, 60)


with open(f"{filepath}/params.txt", "r") as f:
    params = np.array( f.readline().split(","), dtype=np.float32 )




a = 50
eps = 1e-18


def abs_approx( X ):
    return np.sqrt( X*X + eps ) 


def sign_approx( X ):
    ex = np.exp( -a*X )
    return ( 1-ex ) / ( 1+ex )





def scatter_points():
    theta, phi = np.meshgrid( np.linspace( 0, np.pi*0.9, 100 ), np.linspace( -np.pi, np.pi, 100 ), indexing='ij' )
    R = Me25_leaky( params, theta, phi )

    X,Y,Z = spherical_to_cartesian( R, theta, phi, earth_pos )
    
    B_thing = B[np.array(X, dtype=np.int16), np.array(Y, dtype=np.int16), np.array(Z, dtype=np.int16)]
    
    B_thing_norm = np.linalg.norm( B_thing, axis=2 )
    
    saturation = 0.4
    
    B_thing_norm = B_thing_norm * (B_thing_norm < np.max(B_thing_norm)*saturation) + np.max(B_thing_norm)*saturation * (B_thing_norm >= np.max(B_thing_norm)*saturation)
    
    color = (B_thing_norm - np.min(B_thing_norm)) / (np.max(B_thing_norm) - np.min(B_thing_norm))
    cmap = cm.get_cmap("inferno")
    color = cmap( color )
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel=r"$x \in [-30; 128] R_E$", ylabel=r"$y \in [-58; 58] R_E$", zlabel=r"$z \in [-58; 58] R_E$")
    # plt.suptitle("Surface of the magnetopause drawn from the fitted function.")
    # plt.title("Colours represent the norm of the magnetic field.")

    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    
    ax.figure.subplots_adjust(0, 0, 1, 1)

    ax.plot_surface(X, Y, Z, facecolors=color, shade=False)
    
    norm = mpl.colors.Normalize(vmin=np.min(B_thing_norm), vmax=np.max(B_thing_norm))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax = ax, label=r"$||\mathbf{B}||$ [$T$]", shrink=0.5)

    plt.show()






def find_separator():
    nb_theta = 100
    nb_phi = 100
    
    theta, phi = np.meshgrid( np.linspace( 0, np.pi*0.9, nb_theta ), np.linspace( -np.pi, np.pi, nb_phi ), indexing='ij' )
    R = Me25_leaky( params, theta, phi )
    X,Y,Z = spherical_to_cartesian( R, theta, phi, earth_pos )
    
    B_mag = B[np.array(X, dtype=np.int16), np.array(Y, dtype=np.int16), np.array(Z, dtype=np.int16)]
    B_mag_norm = np.linalg.norm( B_mag, axis=2 )
    
    angle_used = 0.3
    offset = int(nb_phi*(0.5-angle_used)//2)
    
    indices_pos = offset + np.argmin( B_mag_norm[:,offset:int(nb_phi*0.5)-offset], axis=1 )
    indices_neg = int(nb_phi*0.5)+offset + np.argmin( B_mag_norm[:,int(nb_phi*0.5)+offset:-offset], axis=1 )
    
    X_sep_pos = np.array( [X[i, indices_pos[i]] for i in range(len(indices_pos))] )
    Y_sep_pos = np.array( [Y[i, indices_pos[i]] for i in range(len(indices_pos))] )
    Z_sep_pos = np.array( [Z[i, indices_pos[i]] for i in range(len(indices_pos))] )
    
    X_sep_neg = np.array( [X[i, indices_neg[i]] for i in range(len(indices_neg))] )
    Y_sep_neg = np.array( [Y[i, indices_neg[i]] for i in range(len(indices_neg))] )
    Z_sep_neg = np.array( [Z[i, indices_neg[i]] for i in range(len(indices_neg))] )
    
    saturation = 0.5
    
    B_mag_norm = B_mag_norm * (B_mag_norm < np.max(B_mag_norm)*saturation) + np.max(B_mag_norm)*saturation * (B_mag_norm >= np.max(B_mag_norm)*saturation)
    
    color = (B_mag_norm - np.min(B_mag_norm)) / (np.max(B_mag_norm) - np.min(B_mag_norm))
    cmap = cm.get_cmap("inferno")
    color = cmap( color )
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set(xlabel=r"$x \in [-30; 128] R_E$", ylabel=r"$y \in [-58; 58] R_E$", zlabel=r"$z \in [-58; 58] R_E$")

    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    
    ax.figure.subplots_adjust(0, 0, 1, 1)

    ax.plot_surface(X, Y, Z, facecolors=color, shade=False, alpha=0.5, linewidth=0)
    
    norm = mpl.colors.Normalize(vmin=np.min(B_mag_norm), vmax=np.max(B_mag_norm))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax = ax, label=r"$||\mathbf{B}||$ [$T$]", shrink=0.5)
    
    ax.scatter( X_sep_pos, Y_sep_pos, Z_sep_pos, color="black", label="Separator" )
    ax.scatter( X_sep_neg, Y_sep_neg, Z_sep_neg, color="black" )
    
    ax.legend()
    
    plt.show()
    
    


def find_current_sheet():
    global J_norm
    global V
    
    if J_norm is None: 
        J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
        
    if V is None: 
        V = import_from(f"{filepath}/V_processed_real.txt")
    V_x = V[:,:,:,0]
    
    nb_theta = 200
    nb_phi = 100
    
    theta, phi = np.meshgrid( np.linspace( 0, np.pi*0.9, nb_theta ), np.linspace( -np.pi, np.pi, nb_phi ), indexing='ij' )
    R = Me25_leaky( params, theta, phi )
    X,Y,Z = spherical_to_cartesian( R, theta, phi, earth_pos )
    
    B_mag = B[np.array(X, dtype=np.int16), np.array(Y, dtype=np.int16), np.array(Z, dtype=np.int16)]
    B_mag_norm = np.linalg.norm( B_mag, axis=2 )
    
    angle_used = 0.3
    offset = int(nb_phi*(0.5-angle_used)//2)
    
    indices_pos = offset + np.argmin( B_mag_norm[:,offset:int(nb_phi*0.5)-offset], axis=1 )
    indices_neg = int(nb_phi*0.5)+offset + np.argmin( B_mag_norm[:,int(nb_phi*0.5)+offset:-offset], axis=1 )
    
    X_sep_pos = np.array( [X[i, indices_pos[i]] for i in range(len(indices_pos))] )
    Y_sep_pos = np.array( [Y[i, indices_pos[i]] for i in range(len(indices_pos))] )
    Z_sep_pos = np.array( [Z[i, indices_pos[i]] for i in range(len(indices_pos))] )
    
    X_sep_neg = np.array( [X[i, indices_neg[i]] for i in range(len(indices_neg))] )
    Y_sep_neg = np.array( [Y[i, indices_neg[i]] for i in range(len(indices_neg))] )
    Z_sep_neg = np.array( [Z[i, indices_neg[i]] for i in range(len(indices_neg))] )
    
    nb_interpolations = 50
    
    T = np.linspace(0,1,nb_interpolations)
    
    X_current_sheet = np.empty( (X_sep_neg.size, nb_interpolations) )
    Y_current_sheet = np.empty( (Y_sep_neg.size, nb_interpolations) )
    Z_current_sheet = np.empty( (Z_sep_neg.size, nb_interpolations) )
    
    for i in range(X_sep_neg.size):
        X_current_sheet[i,:] = X_sep_pos[i] * T + X_sep_neg[i] * (1-T)
        Y_current_sheet[i,:] = Y_sep_pos[i] * T + Y_sep_neg[i] * (1-T)
        Z_current_sheet[i,:] = Z_sep_pos[i] * T + Z_sep_neg[i] * (1-T)
        
    # normal_width = 5
    # nb_normal = 10
    
    # new_X_current_sheet = np.empty( (X_sep_neg.size, nb_interpolations, nb_normal) )
    # new_Y_current_sheet = np.empty( (X_sep_neg.size, nb_interpolations, nb_normal) )
    # new_Z_current_sheet = np.empty( (X_sep_neg.size, nb_interpolations, nb_normal) )

    # ux1 = X_sep_pos - X_sep_neg; uy1 = Y_sep_pos - Y_sep_neg; uz1 = Z_sep_pos - Z_sep_neg
    # u_norm1 = np.sqrt(ux1**2 + uy1**2 + uz1**2)
    # ux1 /= u_norm1; uy1 /=u_norm1; uz1 /= u_norm1
    
    # for i in range(nb_normal):
        
        
    
    new_X_current_sheet = np.copy(X_current_sheet)
    new_Y_current_sheet = np.copy(Y_current_sheet)
    new_Z_current_sheet = np.copy(Z_current_sheet)
        
    _, _, dJ_norm_dz = get_gradients( J_norm[:,:,:,0] )
    
    kernel_size = 3
    dJ_norm_dz = signal.convolve( dJ_norm_dz, np.ones((kernel_size, kernel_size, 1)) / kernel_size**2, 'same' )
    
    dJ_norm_dz /= np.max(dJ_norm_dz)
        
    nb_iterations = 200
        
    for iteration in range(nb_iterations):
        for i in range(X_sep_neg.size):
            for j in range(nb_interpolations):
                
                # if new_X_current_sheet[i,j] < earth_pos[0]: continue
                
                grad = interpolate(
                    [ new_X_current_sheet[i,j], new_Y_current_sheet[i,j], new_Z_current_sheet[i,j] ],
                    dJ_norm_dz
                )
                
                grad_min = 0.5
                grad_max = 5
                
                if np.abs(grad) < grad_min: grad *= grad_min / np.abs(grad)
                elif np.abs(grad) > grad_max: grad *= grad_max / np.abs(grad)
                
                new_Z_current_sheet[i,j] += grad
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set(xlabel=r"$x \in [-30; 128] R_E$", ylabel=r"$y \in [-58; 58] R_E$", zlabel=r"$z \in [-58; 58] R_E$")

    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])

    # dist = np.sqrt(
    #     (new_X_current_sheet - X_current_sheet)**2 + 
    #     (new_Y_current_sheet - Y_current_sheet)**2 + 
    #     (new_Z_current_sheet - Z_current_sheet)**2
    # )
    
    # color = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    
    new_X_current_sheet = new_X_current_sheet.flatten()
    new_Y_current_sheet = new_Y_current_sheet.flatten()
    new_Z_current_sheet = new_Z_current_sheet.flatten()
    
    
    V_x_sheet = V_x[np.array(new_X_current_sheet, dtype=np.int16), np.array(new_Y_current_sheet, dtype=np.int16), np.array(new_Z_current_sheet, dtype=np.int16)]
    
    saturation = 0
    
    V_x_sheet = 0 * (V_x_sheet < saturation) + 1 * (V_x_sheet >= saturation)
    
    
    # J_sheet_norm = J_norm[np.array(new_X_current_sheet, dtype=np.int16), np.array(new_Y_current_sheet, dtype=np.int16), np.array(new_Z_current_sheet, dtype=np.int16)]
    
    # saturation = 5e-9
    
    # J_sheet_norm = J_sheet_norm * (J_sheet_norm < saturation) + saturation * (J_sheet_norm >= saturation)
    
    
    
    color = (V_x_sheet - np.min(V_x_sheet)) / (np.max(V_x_sheet) - np.min(V_x_sheet))
    cmap = plt.get_cmap("inferno")
    color = cmap( color )
    
    ax.scatter( new_X_current_sheet, new_Y_current_sheet, new_Z_current_sheet, c=color, alpha=0.8 )

    norm = mpl.colors.Normalize(vmax=saturation)#=np.max(J_sheet_norm))
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="inferno"),
    #              ax = ax, label=r"$||V_x||$ [$m/s$]", shrink=0.5)


    ax.view_init(elev=0., azim=270.)
    
    plt.savefig("../images/current_sheet_V.svg")
        
    
if __name__=="__main__":  
    find_current_sheet()



import sys


#last line deletion
def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')


def get_animation():
    global J_norm
    
    if J_norm is None: 
        J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")
    
    length = J_norm.shape[1]

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)

    X, Y, Z = np.meshgrid( 
        np.arange( J_norm.shape[0] ),
        np.arange( J_norm.shape[1] ),
        np.arange( J_norm.shape[2] ),
        indexing='ij'
    )
    
    X = earth_pos[0] - X
    Y -= earth_pos[1]
    Z -= earth_pos[2]

    R = np.sqrt( X*X + Y*Y + Z*Z )
    Theta = np.arccos( X / np.maximum(1, R) )
    Phi = np.arccos( Z / np.maximum(1, np.sqrt( Y*Y + Z*Z )) )
    Phi = Phi * (Y>0) - Phi*(Y<=0)
    
    predictedR = Me25_leaky( params, Theta, Phi )
    
    Mask = R <= predictedR
    

    ### vmax
    # B: 1e-7
    # J: 5e-10
    # V: None


    J_xy = axes[0].imshow(J_norm[:,:,0], cmap="inferno", vmin=0, vmax=5e-10)
    plt.colorbar(J_xy, ax=axes[0])
    J_xy = axes[0].imshow(np.ones_like(J_norm[:,:,0]), alpha=0.8*Mask[:,:,0], cmap="Paired")
    axes[0].set_title(fr"$||J|| \in ({0},\hat x, \hat y)$")
    axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")

    J_xz = axes[1].imshow(J_norm[:,0,:], cmap="inferno", vmin=0, vmax=5e-9)
    plt.colorbar(J_xz, ax=axes[1])
    J_xy = axes[1].imshow(np.ones_like(J_norm[:,0,:]), alpha=0.8*Mask[:,0,:], cmap="Paired")
    axes[1].set_title(fr"$||J|| \in ({0},\hat x, \hat z)$")
    axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")


    def animate(i):        
        J_xy = axes[0].imshow(J_norm[:,:,int(i)], cmap="inferno", vmin=0, vmax=5e-10)
        J_xy = axes[0].imshow(np.ones_like(J_norm[:,:,int(i)]), alpha=0.8*Mask[:,:,int(i)], cmap="Paired")
        axes[0].set_title(fr"$||J|| \in ({int(i)},\hat x, \hat y)$")

        J_xz = axes[1].imshow(J_norm[:,int(i),:], cmap="inferno", vmin=0, vmax=5e-9)
        J_xy = axes[1].imshow(np.ones_like(J_norm[:,int(i),:]), alpha=0.8*Mask[:,int(i),:], cmap="Paired")
        axes[1].set_title(fr"$||J|| \in ({int(i)},\hat x, \hat z)$")
        
        delete_last_line()
        print(f"{100*i/length:.2f}% done.")
        
        return J_xy, J_xz
        
    ani = anim.FuncAnimation(fig, animate, interval=150, frames=np.arange(0, length, 2))  
    ani.save(filename=f"fit_vs_reality_{filepath[-10:]}.gif", writer="pillow")
    
    plt.close(fig)


def get_middle():
    global J_norm

    if J_norm is None:
        J_norm = import_from(f"{filepath}/J_norm_processed_real.txt")

    length = J_norm.shape[1]

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)

    X, Y, Z = np.meshgrid(
        np.arange( J_norm.shape[0] ),
        np.arange( J_norm.shape[1] ),
	np.arange( J_norm.shape[2] ),
        indexing='ij'
    )

    X = earth_pos[0] - X
    Y -= earth_pos[1]
    Z -= earth_pos[2]

    R = np.sqrt( X*X + Y*Y + Z*Z )
    Theta = np.arccos( X / np.maximum(1, R) )
    Phi = np.arccos( Z / np.maximum(1, np.sqrt( Y*Y + Z*Z )) )
    Phi = Phi * (Y>0) - Phi*(Y<=0)
    predictedR = Me25( params, Theta, Phi )

    Mask = R <= predictedR


    ### vmax
    # B: 1e-7
    # J: 5e-10
    # V: None


    J_xy = axes[0].imshow(J_norm[:,:,J_norm.shape[2]//2], cmap="inferno", vmin=0, vmax=5e-10)
    plt.colorbar(J_xy, ax=axes[0])
    J_xy = axes[0].imshow(np.ones_like(J_norm[:,:,J_norm.shape[2]//2]), alpha=0.8*Mask[:,:,J_norm.shape[2]//2], cmap="Paired")
    axes[0].set_title(fr"$||J|| \in ({0},\hat x, \hat y)$")
    axes[0].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$y \in [-58; 58] R_E$")

    J_xz = axes[1].imshow(J_norm[:,J_norm.shape[2]//2,:], cmap="inferno", vmin=0, vmax=5e-9)
    plt.colorbar(J_xz, ax=axes[1])
    J_xy = axes[1].imshow(np.ones_like(J_norm[:,J_norm.shape[2]//2,:]), alpha=0.8*Mask[:,J_norm.shape[2]//2,:], cmap="Paired")
    axes[1].set_title(fr"$||J|| \in ({0},\hat x, \hat z)$")
    axes[1].set(ylabel=r"$x \in [-30; 128] R_E$", xlabel=r"$z \in [-58; 58] R_E$")

    plt.show()











# def weird_3d_thing():
#     J = import_from(f"{filepath}/J_processed.txt")
    
#     J_norm = np.linalg.norm(J, axis=3)

#     X, Y, Z = np.meshgrid( 
#         np.arange( J_norm.shape[0] ),
#         np.arange( J_norm.shape[1] ),
#         np.arange( J_norm.shape[2] ),
#         indexing='ij'
#     )
    
#     X = earth_pos[0] - X
#     Y -= earth_pos[1]
#     Z -= earth_pos[2]

#     R = np.sqrt( X*X + Y*Y + Z*Z )
#     Theta = np.arccos( X / np.maximum(1, R) )
#     Phi = np.arccos( Z / np.maximum(1, np.sqrt( Y*Y + Z*Z )) )
#     Phi = Phi * (Y>0) - Phi*(Y<=0)
#     predictedR = Me25( params, Theta, Phi )
    
#     Mask = R <= predictedR  
    
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set(xlabel=r"$x \in [-30; 128] R_E$", ylabel=r"$y \in [-58; 58] R_E$", zlabel=r"$z \in [-58; 58] R_E$")

#     ax.set_xlim(0, shape[0])
#     ax.set_ylim(0, shape[1])
#     ax.set_zlim(0, shape[2])
    
#     for i in range(0, Y.shape[1], 1):
#         pass
        
#     plt.show()
    
      
