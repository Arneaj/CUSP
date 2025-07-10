import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gorgon import import_from, spherical_to_cartesian, get_gradients, interpolate, signal, Me25_leaky
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
    
    
    
    color = (V_x_sheet - np.min(V_x_sheet)) / (np.max(V_x_sheet) - np.min(V_x_sheet))
    cmap = plt.get_cmap("inferno")
    color = cmap( color )
    
    ax.scatter( new_X_current_sheet, new_Y_current_sheet, new_Z_current_sheet, c=color, alpha=0.8 )

    norm = mpl.colors.Normalize(vmax=saturation)#=np.max(J_sheet_norm))
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="inferno"),
    #              ax = ax, label=r"$||V_x||$ [$m/s$]", shrink=0.5)


    ax.view_init(elev=90., azim=270.)
    
    plt.savefig("../images/current_sheet_grid.svg")
        
    
if __name__=="__main__":  
    find_current_sheet()

