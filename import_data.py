from gorgon_tools.magnetosphere import gorgon_import as imp

import numpy as np
import matplotlib.pyplot as plt




import os
import sys
import time


if len(sys.argv) < 3:
    print("No Run path given!")
    exit(1)

run_nb = sys.argv[1]
timestep = sys.argv[2]

filepath = "data/{}_{}".format(run_nb, timestep)


#last line deletion
def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')







sim = imp.gorgon_sim(
	"/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/{}".format(run_nb)
)

index_of_value = np.squeeze( np.where( sim.times == int(timestep) ) )


sim.import_timestep(index_of_value)


sim.import_space("/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/{}/MS/x00_Bvec_c-{}.pvtr".format(run_nb, timestep))



B = sim.arr["Bvec_c"]
J = sim.arr["jvec"]
V = sim.arr["vvec"]

X = sim.xc; Y = sim.yc; Z = sim.zc
# dX = sim.dx; dY = sim.dy; dZ = sim.dz


STEP = 1


with open("{}/X.txt".format(filepath), "w") as f:
    f.write( "{},1,1,1\n".format(X.shape[0]) )

    for ix in range(0, X.shape[0], STEP):
        f.write( str(X[ix]) )
        f.write( "," )

print("finished writing X")

with open("{}/Y.txt".format(filepath), "w") as f:
    f.write( "{},1,1,1\n".format(Y.shape[0]) )

    for ix in range(0, Y.shape[0], STEP):
        f.write( str(Y[ix]) )
        f.write( "," )

print("finished writing Y")


with open("{}/Z.txt".format(filepath), "w") as f:
    f.write( "{},1,1,1\n".format(Z.shape[0]) )

    for ix in range(0, Z.shape[0], STEP):
        f.write( str(Z[ix]) )
        f.write( "," )

print("finished writing Z")


with open("{}/B.txt".format(filepath), "w") as f:
    f.write( "{},{},{},{}\n".format(B.shape[0], B.shape[1], B.shape[2], B.shape[3]) )

    for ix in range(0, B.shape[0], STEP):
        # print( "B: {}%".format(round(100*ix/B.shape[0], 2)) )
        for iy in range(0, B.shape[1], STEP):
            for iz in range(0, B.shape[2], STEP):
                for i in range(3):
                    f.write( str(B[ix,iy,iz,i]) )
                    f.write( "," )
        # delete_last_line()

print("finished writing B")


with open("{}/J.txt".format(filepath), "w") as f:
    f.write( "{},{},{},{}\n".format(J.shape[0], J.shape[1], J.shape[2], J.shape[3]) )
    for ix in range(0, J.shape[0], STEP):
        # print( "J: {}%".format(round(100*ix/J.shape[0], 2)) )
        for iy in range(0, J.shape[1], STEP):
            for iz in range(0, J.shape[2], STEP):
                for i in range(3):
                    f.write( str(J[ix,iy,iz,i]) )
                    f.write( "," )
        # delete_last_line()

print("finished writing J")



with open("{}/V.txt".format(filepath), "w") as f:
    f.write( "{},{},{},{}\n".format(V.shape[0], V.shape[1], V.shape[2], V.shape[3]) )
    for ix in range(0, V.shape[0], STEP):
        # print( "V: {}%".format(round(100*ix/V.shape[0], 2)) )
        for iy in range(0, V.shape[1], STEP):
            for iz in range(0, V.shape[2], STEP):
                for i in range(3):
                    f.write( str(V[ix,iy,iz,i]) )
                    f.write( "," )
        # delete_last_line()

print("finished writing V")
print()



