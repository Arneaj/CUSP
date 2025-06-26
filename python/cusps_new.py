import numpy as np
import matplotlib.pyplot as plt

from gorgon import Me25, import_from, Me25_cusps

import sys


theta = np.linspace(0, np.pi*0.99, 200)

fig, ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(6)


r_0 = 10.4882

alpha_0 = 0.5776
alpha_1 = 0
alpha_2 = 0.1404
e = 0

d = 2
l = 1
s = 0.5

phi = 0
Styles = ['-', '--', '-.']


##### BASIC ONE


r1 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d*1.5, l, s*1.5, d, l, s, e], theta, phi + 0 ) )
r2 = ( Me25( [r_0, alpha_0, alpha_1, alpha_2, d*1.5, l, s*1.5, d, l, s, e], theta, phi + np.pi ) )

X1 = r1 * np.cos(theta)
Z1 = r1 * np.sin(theta)

X2 = r2 * np.cos(theta)
Z2 = r2 * np.sin(theta)

X = np.concatenate( [X2[::-1], X1] )
Z = np.concatenate( [-Z2[::-1], Z1] )

ax.plot( Z, X, '-.', color=(0.2,0.2,0.2), label="Liu12 cusps model" )



##### NEW ONE


r1_bis = ( Me25_cusps( [r_0, alpha_0, alpha_1, alpha_2, d*1.5, l, s*1.5, d, l, s, e, 2, 2], theta, phi + 0 ) )
r2_bis = ( Me25_cusps( [r_0, alpha_0, alpha_1, alpha_2, d*1.5, l, s*1.5, d, l, s, e, 2, 2], theta, phi + np.pi ) )

X1_bis = r1_bis * np.cos(theta)
Z1_bis = r1_bis * np.sin(theta)

X2_bis = r2_bis * np.cos(theta)
Z2_bis = r2_bis * np.sin(theta)

X_bis = np.concatenate( [X2_bis[::-1], X1_bis] )
Z_bis = np.concatenate( [-Z2_bis[::-1], Z1_bis] )

ax.plot( Z_bis, X_bis, '-', color=(0.2,0.2,0.2), label="My cusps model" )





ax.set_xlim(-15, 15)
ax.set_ylim(-10, 15)

ax.set_aspect(0.8)

ax.scatter( [0], [0], marker='x', color=(0.2,0.2,0.2), label="Earth" )

plt.legend()

plt.title( "Liu12 cusps model vs mine\n$\\dfrac{d_N}{d_S} = 1.5$, $\\ell_N = \\ell_S = 0.2$ and $\\dfrac{s_N}{s_S} = 1.5$\n" )
plt.xlabel( "$z$ [$R_E$]" )
plt.ylabel( "$x$ [$R_E$]" )


plt.savefig("cusps_new_l_equals_1aaa.svg")
