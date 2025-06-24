import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from gorgon import import_from, filename, h1_3d, h2_3d, mu_0


B,V,_,rho = import_from(filename)


STEP = 1

B = B[::STEP, ::STEP, ::STEP]
V = V[::STEP, ::STEP, ::STEP]
rho = rho[::STEP, ::STEP, ::STEP]


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


_, Bx_dy, Bx_dz = get_gradients(B[:,:,:,0])
By_dx, _, By_dz = get_gradients(B[:,:,:,1])
Bz_dx, Bz_dy, _ = get_gradients(B[:,:,:,2])







vmax = 5e-1
vmin = 0




J = np.sqrt( (Bz_dy - By_dz)**2 + (Bx_dz - Bz_dx)**2 + (By_dx - Bx_dy)**2 ) / mu_0


plt.subplot(221)
plt.imshow(J[:,80//STEP,:], cmap="inferno", vmax = 5e-1)
plt.colorbar()
plt.title(r"$|\frac{1}{\mu_0} \nabla \times B|$ in $(x,z)$")



plt.subplot(222)
plt.imshow(J[:,:,80//STEP], cmap="inferno", vmax = 5e-1)
plt.colorbar()
plt.title(r"$|\frac{1}{\mu_0} \nabla \times B|$ in $(x,y)$")


_, J_dy, J_dz = get_gradients(J)
divish_J = J_dy + J_dz



plt.subplot(223)
plt.imshow(divish_J[:,80//STEP,:], cmap="inferno", vmin=-2, vmax = 2)
plt.colorbar()
plt.title(r"$(\frac{\partial}{\partial y} + \frac{\partial}{\partial z}) |\frac{1}{\mu_0} \nabla \times B|$ in $(x,z)$")



plt.subplot(224)
plt.imshow(divish_J[:,:,80//STEP], cmap="inferno", vmin=-2, vmax = 2)
plt.colorbar()
plt.title(r"$(\frac{\partial}{\partial y} + \frac{\partial}{\partial z}) |\frac{1}{\mu_0} \nabla \times B|$ in $(x,y)$")
















plt.show()




