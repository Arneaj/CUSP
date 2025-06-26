"""# Library import"""


import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import cm
from matplotlib.colors import Normalize


from scipy import signal

import numpy as np



filename = "../data/Run1"
mu_0 = 1.256637e-6



def import_from(file: str, step: int = None):

    if step is None:
        step = 1

    with open(file) as f:
        lines = f.readlines()
        shape = np.array( lines[0].split(","), dtype=np.int16 )

        vec = np.array( lines[1].split(",")[:-1], dtype=np.float32 ).reshape( shape )[::step,::step,::step]

#    print("finished reading file \"", file, "\"")

    return vec


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



def get_gradients(vector: np.ndarray):
    grad_vector_x = signal.convolve(vector, h2_3d('x'), mode='same')
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('y'), mode='same')
    grad_vector_x = signal.convolve(grad_vector_x, h1_3d('z'), mode='same')

    grad_vector_y = signal.convolve(vector, h1_3d('x'), mode='same')
    grad_vector_y = signal.convolve(grad_vector_y, h2_3d('y'), mode='same')
    grad_vector_y = signal.convolve(grad_vector_y, h1_3d('z'), mode='same')

    grad_vector_z = signal.convolve(vector, h1_3d('x'), mode='same')
    grad_vector_z = signal.convolve(grad_vector_z, h1_3d('y'), mode='same')
    grad_vector_z = signal.convolve(grad_vector_z, h2_3d('z'), mode='same')
    
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
        2 / (1+cos_theta) )**( params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * np.exp( -np.abs(theta - params[5]) / params[6] ) * (np.sign(cos_phi) + 1)/2 +
        params[7] * np.exp( -np.abs(theta - params[8]) / params[9] ) * (np.sign(-cos_phi) + 1)/2
    ) * cos_phi*cos_phi


def Me25_cusps(params: list, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | float:
    """
    Expects theta in [0;pi] and phi in [-pi;pi)
    
    Params are: [ r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e ]
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
        
    is_north = np.abs(phi) <= 0.5 * np.pi
    is_norths_day = theta <= params[5]
    is_souths_day = theta <= params[8]

    return main_part - (
        is_north * params[4] * (
            is_norths_day * ( 1 - np.abs(1-theta*theta/(params[5]*params[5]))**(params[6]*params[11]) ) +
            (1-is_norths_day) * ( np.exp( (params[5] - theta)/params[6] ) )
        ) + 
        (1-is_north) * params[7] * (
            is_souths_day * ( 1 - np.abs(1-theta*theta/(params[8]*params[8]))**(params[9]*params[12]) ) +
            (1-is_souths_day) * ( np.exp( (params[8] - theta)/params[9] ) )
        )
    ) * cos_phi * cos_phi



a = 1
eps = 1e-18


def abs_approx( X ):
    return np.sqrt( X*X + eps )


def sign_approx( X ):
    ex = np.exp( -a*X )
    return ( 1-ex ) / ( 1+ex )


    
def interpolate(P, Bi):
    xm = int(P[0]//1); ym = int(P[1]//1); zm = int(P[2]//1)
    xd = P[0]%1; yd = P[1]%1; zd = P[2]%1

    B3d = Bi[xm:xm+2, ym:ym+2, zm:zm+2]
    B2d = B3d[0]*(1-xd) + B3d[1]*xd
    B1d = B2d[0]*(1-yd) + B2d[1]*yd
    B0d = B1d[0]*(1-zd) + B1d[1]*zd

    return B0d









