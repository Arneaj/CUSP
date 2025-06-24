import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from gorgon import import_from, filename, h2_3d, h1_3d, mu_0

streamlines = []
centerline = []



with open("../data/streamlines.txt", "r") as f:
    lines = f.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
        lines[i] = lines[i].split(",")
        streamlines.append( np.array(lines[i], dtype=np.float32).reshape((-1,3)) )
        
# with open("../data/centerline.txt", "r") as f:
#     line = f.readline()
    
#     line = line.split(",")
#     centerline = np.array(line, dtype=np.float32).reshape((-1,3))





STEP = 1




_,_,J = import_from(filename)

J_norm = np.linalg.norm( J[::STEP,::STEP,::STEP], axis=3 )


fig1 = plt.figure()
ax = fig1.add_subplot(projection="3d", facecolor="black")

ax.set_box_aspect( np.shape( J_norm ) )

ax.set_axis_off()


for streamline in streamlines:
    ax.plot(streamline[:,0], streamline[:,1], streamline[:,2], color=(1,1,1,0.1))



plt.show(block=False)


print(J_norm.shape)



fig2 = plt.figure()
ax1 = fig2.add_subplot(1, 2, 1)
ax1.set_title(r"$||J||$ in $(x,y)$ plane")

ax1.imshow(J_norm[:,:,J_norm.shape[2]//STEP//2], cmap="inferno", vmin=0, vmax=0.5e-9)

for streamline in streamlines:
    ax1.plot(streamline[:,1], streamline[:,0], color=(1,1,1,0.03))


ax2 = fig2.add_subplot(1, 2, 2)
ax2.set_title(r"$||J||$ in $(x,z)$ plane")

ax2.imshow(J_norm[:,J_norm.shape[1]//STEP//2,:], cmap="inferno", vmin=0, vmax=0.5e-9)

for streamline in streamlines:
    ax2.plot(streamline[:,2], streamline[:,0], color=(1,1,1,0.03))

    
plt.show()




