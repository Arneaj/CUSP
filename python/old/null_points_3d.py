import h5py

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import cm

import numpy as np

# import open3d as o3d

# import plotly.graph_objects as go


filename = "Gorgon_20210420__MS_params_7580s.hdf5"



def plot_fit_plane(ax, points):
    U, S, Vh = np.linalg.svd(points)
    
    H = Vh[-1]
    
    x_plane = np.linspace(0, 240, 2)
    y_plane = np.linspace(0, 160, 2)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = -(H[0]*X_plane + H[1]*Y_plane + H[3]) / H[2]
    
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5)


# def display_inlier_outlier(cloud, ind):
#     inlier_cloud = cloud.select_by_index(ind)
#     outlier_cloud = cloud.select_by_index(ind, invert=True)

#     print("Showing outliers (red) and inliers (gray): ")
#     outlier_cloud.paint_uniform_color([1, 0, 0])
#     inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                       zoom=0.3412,
#                                       front=[0,-1,0],
#                                       lookat=[0,0,0],
#                                       up=[0,0,-1],
#                                       width=1024, height=512)


def isosurface_plot():
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
    
    STEP = 1

    B = B[::STEP, ::STEP, ::STEP]
    B_norm = np.linalg.norm(B, axis=3)

    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.set_xlim(0, B.shape[0])
    ax.set_ylim(0, B.shape[1])
    ax.set_zlim(0, B.shape[2])

    X, Y, Z = np.indices(B.shape[:-1])
    
    LVL = 0
    
    mask = np.abs(B_norm - LVL) < 1e-9#1.942125e-9
    X_mask, Y_mask, Z_mask = (X[mask], Y[mask], Z[mask])
    B_norm_mask = B_norm[mask]
    
    # points = np.stack([X_mask, Y_mask, Z_mask], axis=-1)#, np.ones_like(X_mask)], axis=-1)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.io.write_point_cloud("TestData/sync.ply", pcd)


    # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("TestData/sync.ply")
    
    # pcd_load.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals

    # pcd_load.estimate_normals()
    
    # cl, ind = pcd_load.remove_radius_outlier(nb_points=5, radius=20)
    # display_inlier_outlier(pcd_load, ind)
    
    
    if np.sum(mask) > 0:
        ax.scatter(
            X_mask, Y_mask, Z_mask, 
            alpha=0.1, 
            c=B_norm_mask, cmap=cm.get_cmap("binary")
        )

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show() 




if __name__ == "__main__":
    isosurface_plot()

