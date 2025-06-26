import numpy as np
import matplotlib.pyplot as plt

from gorgon import Me25, import_from

import sys


theta = np.linspace(0, np.pi*0.99, 100)

fig, ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(6)


r_0 = 10.4882

alpha_0 = 0.5776
alpha_1 = 0
alpha_2 = 0.1404
e = 0

d = 3
l = 1.1574
s = 1

Phi = [0, np.pi/4, np.pi/2]
Styles = ['-', '--', '-.']

for i in range(len(Phi)):
    phi = Phi[i]
    style = Styles[i]
    
    r1 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d, l, s*2, d, l, s, e], theta, phi + 0 ) )
    r2 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d, l, s*2, d, l, s, e], theta, phi + np.pi ) )

    X1 = r1 * np.cos(theta)
    Z1 = r1 * np.sin(theta)

    X2 = r2 * np.cos(theta)
    Z2 = r2 * np.sin(theta)
    
    X = np.concatenate( [X2[::-1], X1] )
    Z = np.concatenate( [-Z2[::-1], Z1] )

    ax.plot( Z, X, style, color=(0.2,0.2,0.2), label=rf"$\phi = {phi:.2f}$" )


ax.set_xlim(-15, 15)
ax.set_ylim(-10, 15)

ax.set_aspect(0.8)

ax.scatter( [0], [0], marker='x', color=(0.2,0.2,0.2), label="Earth" )

ax.plot( [0, 0], [-10, 15], '--', color=(0.5,0.5,0.5, 0.2), label="Point $\\theta = 0$" )

plt.legend()

plt.title( "Liu12 cusps model in different planes containing the $\\hat x$-axis rotated by $\\phi$\n$d_N = d_S$, $\\ell_N = \\ell_S$ and $\\dfrac{s_N}{s_S} = 2$\n" )
plt.xlabel( "[$R_E$]" )
plt.ylabel( "$x$ [$R_E$]" )


plt.savefig("cusps.svg")
