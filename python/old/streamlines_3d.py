import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gorgon import filename, import_from

"""# Functions

## Function: interpolate()
"""

def interpolate(P, Bi):
    xm = int(P[0]//1); ym = int(P[1]//1); zm = int(P[2]//1)
    xd = P[0]%1; yd = P[1]%1; zd = P[2]%1

    B3d = Bi[xm:xm+2, ym:ym+2, zm:zm+2]
    B2d = B3d[0]*(1-xd) + B3d[1]*xd
    B1d = B2d[0]*(1-yd) + B2d[1]*yd
    B0d = B1d[0]*(1-zd) + B1d[1]*zd

    return B0d


"""## Function: RK4()"""

def norm_RK4(P_init, Bx,By,Bz, step):
    P = P_init
    k = np.zeros((4))

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k1 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])
    k1 /= np.linalg.norm(k1)

    P = P_init + step*k1/2

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k2 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])
    k2 /= np.linalg.norm(k2)

    P = P_init + step*k2/2

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k3 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])
    k3 /= np.linalg.norm(k3)

    P = P_init + step*k3

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k4 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])
    k4 /= np.linalg.norm(k4)

    P = P_init + step*(k1+2*k2+2*k3+k4)/6

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    return P

def RK4(P_init, Bx,By,Bz, step):
    P = P_init
    k = np.zeros((4))

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k1 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])

    P = P_init + step*k1/2

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k2 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])

    P = P_init + step*k2/2

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k3 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])

    P = P_init + step*k3

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    k4 = np.array([interpolate(P, Bx),
                  interpolate(P, By),
                  interpolate(P, Bz)])

    P = P_init + step*(k1+2*k2+2*k3+k4)/6

    if (P[0]>=Bx.shape[0]) or (P[0]<0): raise
    if (P[1]>=Bx.shape[0]) or (P[1]<0): raise
    if (P[2]>=Bx.shape[0]) or (P[2]<0): raise

    return P


"""## Function: find_streamline()"""

def find_streamline(Bx,By,Bz, x_i,y_i,z_i, step, max_length=1000):
    P = np.array([x_i, y_i, z_i])
    forward_points = [P]
    t = 0

    while t < max_length:
        try:
            P = norm_RK4(P, Bx,By,Bz, step)
        except:
            break

        forward_points.append(P)

        t += 1
        
    t = 0
    P = np.array([x_i, y_i, z_i])
    backward_points = []
    m_Bx, m_By, m_Bz = -Bx,-By,-Bz
    while t < max_length:
        try:
            P = norm_RK4(P, m_Bx,m_By,m_Bz, step)
        except:
            break

        backward_points.insert(0, P)

        t += 1

    return np.array(backward_points + forward_points, dtype=np.int16)


"""## Function: plot_streamlines()"""

def plot_streamlines_nb(axis, Bx,By,Bz, nb_x,nb_y,nb_z, step, max_length=1000):
    for ix in np.linspace(0,Bx.shape[0]-1, nb_x):
        for iy in np.linspace(0,Bx.shape[1]-1, nb_y):
            for iz in np.linspace(0,Bx.shape[2]-1, nb_z):
                points = find_streamline(Bx,By,Bz, ix,iy,iz, step, max_length)
                axis.plot(points[:,0], points[:,1], points[:,2], color=(1,1,1,0.4))

def plot_streamlines_pos(axis, Bx,By,Bz, positions, step, max_length=1000):
    for position in positions:
        points = find_streamline(Bx,By,Bz, position[0],position[1],position[2], step, max_length)
        axis.plot(points[:,0], points[:,1], points[:,2], color=(1,1,1,0.2))


"""## Function: smooth_step()"""

def smooth_interpolation(points, nb_interpolation):    
    if points.shape[0]-nb_interpolation < 0:
        return points

    J = np.ones((nb_interpolation)) / nb_interpolation

    smooth_points = np.empty( (points.shape[0]-nb_interpolation+1, 3) )
    
    # print("points: ", points[:,0].shape)
    # print("J: ", J.shape)
    # print("smooth points: ", smooth_points[:,0].shape)
    # print()
    
    for i in range(3): 
        smooth_points[:,i] = signal.convolve(points[:,i], J, "valid")
        
    return smooth_points

"""## Function: plot_smooth_streamlines()"""

def plot_smooth_streamlines_pos(axis, Bx,By,Bz, positions, step, nb_interpolation=5, max_length=1000):
    for position in positions:
        points = find_streamline(Bx,By,Bz, position[0],position[1],position[2], step, max_length)
        smooth_points = smooth_interpolation(points, nb_interpolation)        
        axis.plot(smooth_points[:,0], smooth_points[:,1], smooth_points[:,2], color=(1,1,1,0.2))


"""# Main function"""

def main():
    """## Read file and extract values"""

    B,V,T,rho = import_from(filename)

    """## Variables"""

    # XMAX = 126
    # XMIN = 30

    # YMAX = 135
    # YMIN = 25

    # ZMAX = 106
    # ZMIN = 54
    
    XMAX = 240
    XMIN = 30

    YMAX = 160
    YMIN = 0

    ZMAX = 160
    ZMIN = 0
    
    STEP = 1

    # x = np.arange(0, XMAX-XMIN)
    # y = np.arange(0, YMAX-YMIN)
    # z = np.arange(0, ZMAX-ZMIN)

    # X,Y,Z = np.meshgrid(x,y,z)

    # B_norm = np.linalg.norm(B[XMIN:XMAX, YMIN:YMAX, ZMIN:ZMAX], axis=3)
    B_x = V[XMIN:XMAX:STEP, YMIN:YMAX:STEP, ZMIN:ZMAX:STEP, 0]
    B_y = V[XMIN:XMAX:STEP, YMIN:YMAX:STEP, ZMIN:ZMAX:STEP, 1]
    B_z = V[XMIN:XMAX:STEP, YMIN:YMAX:STEP, ZMIN:ZMAX:STEP, 2]
    
    """## Plotting"""

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', facecolor='black')

    ax.set_xlim(0, B_x.shape[0])
    ax.set_ylim(0, B_y.shape[1])
    ax.set_zlim(0, B_z.shape[2])
    ax.set_axis_off()
    
    # base_pos = np.array([ (60-XMIN)//STEP, (YMAX-YMIN)//2//STEP, (ZMAX-ZMIN)//2//STEP ])
    # positions = []
    # NB_POINTS = 5
    # RADIUS = 15
    # for ix in range(NB_POINTS):
    #     for iy in range(NB_POINTS):
    #         for iz in range(NB_POINTS):
    #             positions.append( base_pos + np.array([ix,iy,iz]) * 2*RADIUS/(NB_POINTS-1)/STEP - RADIUS/STEP )
    
    
    positions = []
    # DENSITY = 20
    # for ix in np.linspace(XMIN+1, XMAX-2, DENSITY):
    #     for iy in np.linspace(YMIN+1, YMAX-2, DENSITY):
    #             positions.append( np.array([ix,iy,ZMIN+1]) )
    #             positions.append( np.array([ix,iy,ZMAX-2]) )
    
    DENSITY = 20
    for iy in np.linspace(YMIN+1, YMAX-2, DENSITY):
        for iz in np.linspace(ZMIN+1, ZMAX-2, DENSITY):
                positions.append( np.array([XMIN+1,iy,iz]) )
    
    

    # plot_streamlines_nb(ax, B_x,B_y,B_z, 5,5,5, 1)
    # plot_streamlines_pos(ax, B_x,B_y,B_z, positions, 1)
    # plt.show()
    
    plot_smooth_streamlines_pos(ax, B_x,B_y,B_z, positions, 1)
    plt.show()

if __name__ == "__main__":
    main()
