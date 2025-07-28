"""# Library import"""


import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import cm
from matplotlib.colors import Normalize


from scipy import signal

import numpy as np

import struct



mu_0 = 1.256637e-6

import sys


#last line deletion
def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')



def import_from(file: str, step: int = None):
    with open(file) as f:
        lines = f.readlines()
        shape = np.array( lines[0].split(","), dtype=np.int16 )

        vec = np.array( lines[1].split(",")[:-1], dtype=np.float32 ).reshape( shape )

    return vec


def import_from_bin(file: str):
    with open(file, 'rb') as f:
        magic, type_size, x_dim, y_dim, z_dim, i_dim = struct.unpack('IIIIII', f.read(24))
        
        if magic != 0x12345678:
            print(f"{file} is corrupted!")
            exit(1)
        
        f.seek(40) 
        
        dtype = np.float32 if type_size == 4 else np.float64
        data = np.frombuffer(f.read(), dtype=dtype).reshape((x_dim, y_dim, z_dim, i_dim), order='F')

    return data


def grad_mag_angle(vector: np.ndarray):
    scharr = np.array([ 
                        [ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]
                      ]) # Gx + j*Gy
    
    grad_vector = signal.convolve2d(vector, scharr, boundary='symm', mode='same')
    grad_vector_mag = np.abs( grad_vector )
    grad_vector_angle = np.angle( grad_vector )
    
    return grad_vector_mag, grad_vector_angle


def gaussian_kernel_2d(size:int, std:float):
    X,Y = np.indices((size, size))
    return np.exp( (X**2 + Y**2)/(2*std*std) ) / (2*np.pi*std*std)



def h2_3d(dir: str) -> np.ndarray:
    if dir == 'z':
        return np.array( [[[1,0,-1]]] )
    if dir == 'y':
        return np.array( [[[1],[0],[-1]]] )
    if dir == 'x':
        return np.array( [[[1]],[[0]],[[-1]]] )
    else: 
        exit(1)

def h1_3d(dir: str) -> np.ndarray:
    if dir == 'z':
        return np.array( [[[1,2,1]]] )
    if dir == 'y':
        return np.array( [[[1],[2],[1]]] )
    if dir == 'x':
        return np.array( [[[1]],[[2]],[[1]]] )
    else: 
        exit(1)


def grad_mag_3d(vector: np.ndarray) -> np.ndarray:
    grad_vector_x = signal.convolve(vector, h2_3d('x'), mode='same')
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('y'), mode='same')
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('z'), mode='same')
    
    grad_vector_y = signal.convolve(vector, h1_3d('x'), mode='same')
    grad_vector_y = signal.convolve(grad_vector_y, h2_3d('y'), mode='same')
    grad_vector_y = signal.convolve(grad_vector_y, h1_3d('z'), mode='same')
    
    grad_vector_z = signal.convolve(vector, h1_3d('x'), mode='same')
    grad_vector_z = signal.convolve(grad_vector_z, h1_3d('y'), mode='same')
    grad_vector_z = signal.convolve(grad_vector_z, h2_3d('z'), mode='same')
    
    grad_vector_mag = np.sqrt( grad_vector_x**2 + grad_vector_y**2 + grad_vector_z**2 )
    
    return grad_vector_mag


def gaussian_kernel_3d(size:int, std:float) -> np.ndarray:
    X,Y,Z = np.indices((size, size, size))
    return np.exp( (X**2 + Y**2 + Z**2)/(2*std*std) ) / (np.sqrt(2*np.pi*std)**3)



def get_gradients(vector: np.ndarray, mode="same"):
    grad_vector_x = signal.convolve(vector, h2_3d('x'), mode=mode)
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('y'), mode=mode)
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('z'), mode=mode)

    grad_vector_y = signal.convolve(vector, h1_3d('x'), mode=mode)
    grad_vector_y = signal.convolve(grad_vector_y, h2_3d('y'), mode=mode)
    grad_vector_y = signal.convolve(grad_vector_y, h1_3d('z'), mode=mode)

    grad_vector_z = signal.convolve(vector, h1_3d('x'), mode=mode)
    grad_vector_z = signal.convolve(grad_vector_z, h1_3d('y'), mode=mode)
    grad_vector_z = signal.convolve(grad_vector_z, h2_3d('z'), mode=mode)
    
    return grad_vector_x, grad_vector_y, grad_vector_z




def spherical_to_cartesian( R, theta, phi, earth_pos ):
    X = earth_pos[0] - R * np.cos(theta)
    Y = earth_pos[1] + R * np.sin(theta) * np.sin(phi)
    Z = earth_pos[2] + R * np.sin(theta) * np.cos(phi)
    return X,Y,Z






def Shue97(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [-pi;pi) and phi in [0;pi)
    """
    
    cos_theta = np.cos(theta)
    
    return params[0] * ( 2 / (1+cos_theta) )**( params[1] )


def Liu12(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    """

    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)

    return params[0] * ( 
        2 / (1+cos_theta) )**( params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * np.exp( -np.abs(theta - params[5]) / params[6] ) * (np.sign(cos_phi) + 1)/2 +
        params[7] * np.exp( -np.abs(theta - params[8]) / params[9] ) * (np.sign(-cos_phi) + 1)/2
    ) * cos_phi*cos_phi


def Me25(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    """

    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)

    return params[0] * (
        (1 + params[10]) / (1 + params[10]*cos_theta)
    ) * (
        2 / (1+cos_theta) 
    )**( 
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * np.exp( -np.abs(theta - params[5]) / params[6] ) * (np.sign(cos_phi) + 1)/2 +
        params[7] * np.exp( -np.abs(theta - params[8]) / params[9] ) * (np.sign(-cos_phi) + 1)/2
    ) * cos_phi*cos_phi



def sigmoid( x: np.ndarray | float, v: float = 5 ):
    return 1 / ( 1 + np.exp( -v*x ) )



def Me25_cusps(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    
    Params are: [ r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e, a_n, a_s ]
    """

    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    
    main_part = params[0] * ( 
        (1+params[10])/(1+params[10]*cos_theta) 
    ) * (
        2 / (1+cos_theta)
    ) ** (
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    )
        
    day_N = 1 - np.abs(1 - theta**2 / params[5]**2)**(params[6]*params[11])
    night_N = np.exp( -np.abs( params[5] - theta ) / params[6] )
    
    day_S = 1 - np.abs(1 - theta**2 / params[8]**2)**(params[9]*params[12])
    night_S = np.exp( -np.abs( params[8] - theta ) / params[9] )

    return main_part - ( 
        params[4] * sigmoid(np.pi/2 + phi) * sigmoid(np.pi/2 - phi) * (
            day_N * sigmoid(theta) * sigmoid(-theta-params[5]) +
            night_N * sigmoid(theta-params[5])
        ) + 
        params[7] * (sigmoid(-np.pi/2 + phi) + sigmoid(-np.pi/2 - phi)) * (
            day_S * sigmoid(theta) * sigmoid(-theta-params[8]) +
            night_S * sigmoid(theta-params[8])
        )
    )*cos_phi*cos_phi



def l_abs(X: np.ndarray | float) -> np.ndarray | float:
    pos = (X >= 0)
    return pos*X - (1-pos)*0.01*X


def Me25_leaky(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [-pi;pi) and phi in [-pi/2;pi/2)
    
    Params are: [ r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e, a_n, a_s ]
    """

    cos_phi = np.cos(phi)
    cos_theta = np.maximum( np.cos(theta), -0.9999 )
    
    main_part = params[0] * ( 
        (1+params[10])/(1+params[10]*cos_theta) 
    ) * (
        2 / (1+cos_theta)
    ) ** (
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    )
    
    leaky_n = np.maximum( l_abs(theta), 0.01 )
    cusp_n = params[4] * np.exp( -np.abs( leaky_n**params[11] - params[5] * leaky_n**(params[11]-1) ) / params[6] )
    
    leaky_s = np.maximum( l_abs(-theta), 0.01 )
    cusp_s = params[7] * np.exp( -np.abs( leaky_s**params[12] - params[8] * leaky_s**(params[12]-1) ) / params[9] )
    
    return main_part - (cusp_n+cusp_s)*cos_phi*cos_phi




    
def interpolate(P, Bi):
    xm = int(P[0]//1); ym = int(P[1]//1); zm = int(P[2]//1)
    xd = P[0]%1; yd = P[1]%1; zd = P[2]%1

    B3d = Bi[xm:xm+2, ym:ym+2, zm:zm+2]
    B2d = B3d[0]*(1-xd) + B3d[1]*xd
    B1d = B2d[0]*(1-yd) + B2d[1]*yd
    B0d = B1d[0]*(1-zd) + B1d[1]*zd

    return B0d






def Me25_fix(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    
    - params[0]: r_0
    - params[1]: alpha_0 in [0, 1]
    - params[2]: alpha_1
    - params[3]: alpha_2
    - params[4]: d_n in [0, +inf]
    - params[5]: l_n in [0, pi]
    - params[6]: s_n in [0, +inf]
    - params[7]: d_s in [0, +inf]
    - params[8]: l_s in [0, pi]
    - params[9]: s_s in [0, +inf]
    - params[10]: e in [-1, 1]
    - params[11]: a_n in [0, 1]
    - params[12]: a_s in [0, 1]
    """
    
    near_zero = (np.abs(theta) < 1e-3)
    
    theta = theta*(1-near_zero) + 1e-3*near_zero

    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    
    main_part = params[0] * ( 
        (1+params[10])/(1+params[10]*cos_theta) 
    ) * (
        2 / (1+cos_theta)
    ) ** (
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    )
        
    north = np.exp( -np.abs( theta**(params[11]) - params[5]*theta**(params[11]-1) ) / params[6] )
    south = np.exp( -np.abs( theta**(params[12]) - params[8]*theta**(params[12]-1) ) / params[9] )

    return main_part - ( 
        params[4] * sigmoid(np.pi/2 + phi) * sigmoid(np.pi/2 - phi) * north + 
        params[7] * (sigmoid(-np.pi/2 + phi) + sigmoid(-np.pi/2 - phi)) * south
    )*cos_phi*cos_phi




def Me25_poly(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    
    - params[0]: r_0 in [0, +inf]
    - params[1]: alpha_0 in [0, 1]
    - params[2]: alpha_1 in [-1, 1]
    - params[3]: alpha_2 in [-1, 1]
    - params[4]: d_n in [0, +inf]
    - params[5]: l_n in [0, pi]
    - params[6]: s_n in [0, +inf]
    - params[7]: d_s in [0, +inf]
    - params[8]: l_s in [0, pi]
    - params[9]: s_s in [0, +inf]
    - params[10]: e in [-1, 1]
    """
    

    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    
    main_part = params[0] * ( 
        (1+params[10])/(1+params[10]*cos_theta) 
    ) * (
        2 / (1+cos_theta)
    ) ** (
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    )
    
    theta_by_ln_to_sn = (theta / params[5])**params[6]
    theta_by_ls_to_ss = (theta / params[8])**params[9]
        
    north = np.abs((1.0-theta_by_ln_to_sn)/(1.0+theta_by_ln_to_sn )) - 1.0
    south = np.abs((1.0-theta_by_ls_to_ss)/(1.0+theta_by_ls_to_ss )) - 1.0
    
    sigm_cos_phi = sigmoid( np.cos(phi) )

    return main_part + ( 
        params[4] * north * sigm_cos_phi + 
        params[7] * south * (1.0 - sigm_cos_phi)
    )*cos_phi*cos_phi


