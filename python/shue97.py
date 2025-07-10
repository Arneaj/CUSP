import numpy as np
import matplotlib.pyplot as plt

from gorgon import Me25


theta = np.linspace(-np.pi*0.999999, np.pi*0.999999, 200)

r_0 = 10

Alpha = np.linspace(0, 0.9, 7)
E = np.array([ -0.5, 0, 0.3, 0.7 ]) #np.linspace(-0.5, 0.7, 4)

fig, axes = plt.subplots(E.size//2, E.size//2)
fig.set_figwidth(8)
fig.set_figheight(7)

for ie in range(E.size):
    for ia in range(Alpha.size):
        alpha = Alpha[ia]
        e = E[ie]
        
        r = ( Me25( [r_0, alpha, 0, 0, 0, 0, 1, 0, 0, 1, e], theta, 0 ) )
        
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        
        axes[ie//2, ie%2].plot( X, Y )
        
    axes[ie//2, ie%2].set_xlim(-100, 20)
    axes[ie//2, ie%2].set_ylim(-60, 60)
    axes[ie//2, ie%2].set_aspect(1)
    if (abs(e) < 0.01): axes[ie//2, ie%2].set_title( rf"$e = {e:.2f}$ (Shue97)" )
    else: axes[ie//2, ie%2].set_title( rf"$e = {e:.2f}$" )
    
    axes[ie//2, ie%2].scatter( [0], [0] )

fig.suptitle( r"$r = r_{0} \cdot \dfrac{1+e}{1+e \cdot \cos \theta} \left(  \dfrac{2}{1 + \cos \theta}\right)^{\alpha}$ for $\alpha$ going from " + f"${np.min(Alpha):.2f}$ (blue) to ${np.max(Alpha):.2f}$ (pink)" )

plt.savefig("../images/shue97.svg")
