import sys
import numpy as np

if len(sys.argv) < 2:
    print("No Run path given!")
    exit(1)

filepath = sys.argv[1]

earth_pos = [30.75, 58, 58]

interest_points_theta = []
interest_points_phi = []
interest_points_r = []
interest_points_w = []

with open(f"{filepath}/interest_points_cpp.csv", "r") as f:
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
    


X = earth_pos[0] + interest_points_r * np.cos(interest_points_theta)
Y = earth_pos[1] + interest_points_r * np.sin(interest_points_theta) * np.sin(interest_points_phi)
Z = earth_pos[2] + interest_points_r * np.sin(interest_points_theta) * np.cos(interest_points_phi)



nb_phi = 0
theta_0 = interest_points_theta[0]

while np.abs(interest_points_theta[nb_phi] - theta_0) < 1e-3:
    nb_phi += 1
    
nb_theta = interest_points_phi.size // nb_phi
    
X = X.reshape( (nb_theta, nb_phi) )
D = np.sqrt( Y*Y + Z*Z ).reshape( (nb_theta, nb_phi) )
W = interest_points_w.reshape( (nb_theta, nb_phi) )

RADIUS_FOR_AVG = 3

X = np.concatenate( [ X[:,:RADIUS_FOR_AVG], X[:,-RADIUS_FOR_AVG:], X[:,nb_phi//2-RADIUS_FOR_AVG:nb_phi//2+RADIUS_FOR_AVG] ], axis=1 )
W = np.concatenate( [ W[:,:RADIUS_FOR_AVG], W[:,-RADIUS_FOR_AVG:], W[:,nb_phi//2-RADIUS_FOR_AVG:nb_phi//2+RADIUS_FOR_AVG] ], axis=1 )


avg_X = np.sum( X*W, axis=1 ) / np.sum( W, axis=1 )



d_avg_X = avg_X - np.median( np.sort( avg_X )[-RADIUS_FOR_AVG:] )


THRESHOLD = 2
nb_low_var = 0

for i,v in enumerate(d_avg_X):
    if np.abs(v) < THRESHOLD:
        nb_low_var = i

print( f"Before the average P.x for some theta permanently moved more than {THRESHOLD}:" )
print( f"\t--> theta reached {interest_points_theta[nb_phi*nb_low_var]}" )




