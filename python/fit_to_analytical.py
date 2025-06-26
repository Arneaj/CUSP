import numpy as np
from scipy.optimize import least_squares

from gorgon import import_from, Me25, Me25_cusps
from earth_pos_detection import get_earth_pos


import sys

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]


a = 50
eps = 1e-18


def abs_approx( X ):
    return np.sqrt( X*X + eps )


def sign_approx( X ):
    ex = np.exp( -a*X )
    return ( 1-ex ) / ( 1+ex )





interest_points_theta = []
interest_points_phi = []
interest_points_r = []
interest_points_w = []



with open(f"{filepath}/interest_points_cpp.txt", "r") as f:
    lines = f.readlines()

    for line in lines:
        point = np.array( line.split(","), dtype=np.float32 )
        interest_points_theta.append( point[0] )
        interest_points_phi.append( point[1] )
        interest_points_r.append( point[2] )
        interest_points_w.append( point[3] )

    interest_points_theta = np.array(interest_points_theta)
    interest_points_phi = np.array(interest_points_phi)
    interest_points_r = np.array(interest_points_r)
    interest_points_w = np.array(interest_points_w)

# theta_mask = np.abs(interest_points_theta) > 0.5
# interest_points_theta = interest_points_theta[theta_mask]
# interest_points_phi = interest_points_phi[theta_mask]
# interest_points_r = interest_points_r[theta_mask]
# interest_points_w = interest_points_w[theta_mask]


nb_phi = 0

for theta in interest_points_theta:
    if np.abs(theta - interest_points_theta[0]) > 1e-3:
        break
    nb_phi += 1

nb_theta = np.size( interest_points_theta ) // nb_phi

#print("nb_theta: ", nb_theta)
#print("nb_phi: ", nb_phi)


B = import_from(f"{filepath}/B_processed_sim.txt")
B_norm = np.linalg.norm(B, axis=3)

X = import_from(f"{filepath}/X.txt")
Y = import_from(f"{filepath}/Y.txt")
Z = import_from(f"{filepath}/Z.txt")

shape = B_norm.shape
earth_pos_tilde = get_earth_pos(B_norm)

x_min = np.array([np.min(X), np.min(Y), np.min(Z)])
x_max = np.array([np.max(X), np.max(Y), np.max(Z)])
dx = x_max - x_min

#print(f"Simulation shape: {shape}")
#print(f"Earth position: {earth_pos_tilde}")
#print(f"(x,y,z) ranges: {dx}")

x_tilde = earth_pos_tilde[0] + interest_points_r * np.cos(interest_points_theta)
y_tilde = earth_pos_tilde[1] + interest_points_r * np.sin(interest_points_theta) * np.sin(interest_points_phi)
z_tilde = earth_pos_tilde[2] + interest_points_r * np.sin(interest_points_theta) * np.cos(interest_points_phi)

X = x_tilde * dx[0] / shape[0] + x_min[0]
Y = y_tilde * dx[1] / shape[1] + x_min[1]
Z = z_tilde * dx[2] / shape[2] + x_min[2]

interest_points_r = np.sqrt( X*X + Y*Y + Z*Z )
interest_points_theta = np.arccos( X / np.maximum(1, interest_points_r) )
interest_points_phi = np.arccos( Z / np.maximum(1, np.sqrt( Y*Y + Z*Z )) )
interest_points_phi = interest_points_phi * (Y>0) - interest_points_phi*(Y<=0)





theta = interest_points_theta.reshape( (nb_theta, nb_phi) )[4:-4]
phi = interest_points_phi.reshape( (nb_theta, nb_phi) )[4:-4]
radii = interest_points_r.reshape( (nb_theta, nb_phi) )[4:-4]
weights = interest_points_w.reshape( (nb_theta, nb_phi) )[4:-4]

mean_rad = np.mean(radii)

scaled_radii = radii / mean_rad


def residual_function(params, theta_grid, phi_grid, observed_radii, analytical_func, weights):
    """
    params: 10-element array of hyperparameters
    theta_grid, phi_grid: NxM arrays of angle coordinates
    observed_radii: NxM array of measured radii
    analytical_func: function that takes (params, theta, phi) and returns radius
    """
    # Compute analytical radii at all grid points
    predicted_radii = analytical_func(params, theta_grid, phi_grid)

    scaled_radii = predicted_radii / mean_rad

    # Return flattened residuals
    residuals = ((observed_radii - scaled_radii)*weights).flatten()
    return residuals


methods_to_try = ['trf']#, 'dogbox']
results = []
costs = []

for i in range(5):
    for method in methods_to_try:
        initial_guess = np.random.uniform(0, 1, 11)
        #initial_guess *= np.array( [20, 0, 0, 0, 0, 2*np.pi, 10,  10, 2*np.pi,  10 ] )
                    #              r_0, alpha_0, alpha_1, alpha_2, d_n,        l_n,  s_n, d_s,      l_s,     s_s     e  a_n  a_s
        initial_guess *= np.array( [20,       1,       1,       1,  10,    np.pi/2,  0.9,  10,   np.pi/2,    0.9,  1.4,   2,   2 ] )
        initial_guess += np.array( [ 0,       0,       0,       0,   0,          0, 0.01,   0,         0,   0.01, -0.7, 0.1, 0.1 ] )

        results.append(least_squares(
            residual_function,
            initial_guess,
            args=(theta, phi, scaled_radii, Me25_cusps, weights),
            method=method,
            ftol=1e-8,      # Stricter tolerance
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=10000,  # More function evaluations
            #           r_0, alpha_0, alpha_1, alpha_2, d_n,      l_n,  s_n,    d_s,      l_s,   s_s    e    a_n   a_s
            bounds=([     0,       0,       0,       0,   0,        0, 0.01,      0,        0, 0.01, -0.95, 0.01, 0.01], 
                    [np.inf,       1,       1,       1,  15,  np.pi/2,    1,     15,  np.pi/2,    1,  0.95,    3,    3])
        ))
#        print(f"Method {method}: cost={results[-1].cost:.6f}, success={results[-1].success}")
        costs.append(results[-1].cost)


best_index = np.argmin( costs )

# print(f"Best method: {methods_to_try[best_index]}. Parameters: {results[best_index].x}")

print(f"Lowest cost: {costs[best_index]:.3f} with parameters: {results[best_index].x}")


with open(f"{filepath}/params.txt", "w") as f:
    for i in range(len(results[best_index].x)):
        if i > 0: f.write(',')
        f.write( str( (results[best_index].x)[i] ) )



#  1.14219956e+01  5.92777369e-01  3.05147063e-03  4.59252038e-02
#  6.42790224e+00  5.22724908e-01  1.34866949e-01  6.51839827e+00
# -5.14979693e-01  1.35549188e-01





