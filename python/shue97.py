import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from gorgon import Me25


theta = np.linspace(-np.pi*0.999999, np.pi*0.999999, 300)

r_0 = 10

Alpha = np.array([0.2, 0.4, 0.6, 0.8])
E = np.array([ -0.3, 0, 0.3, 0.6 ]) #np.linspace(-0.5, 0.7, 4)
Styles = ["-", "--", "-.", ":"]

fig, axes = plt.subplots(3, 2, gridspec_kw={"height_ratios":[1, 50, 50]})
fig.set_figwidth(8)
fig.set_figheight(7)

gridspec = axes[0, 0].get_subplotspec().get_gridspec()

for a in axes[0, :]:
    a.remove()

for ie in range(E.size):
    for ia in range(Alpha.size):
        alpha = Alpha[ia]
        e = E[ie]
        
        r = ( Me25( [r_0, alpha, 0, 0, 0, 0, 1, 0, 0, 1, e], theta, 0 ) )
        
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        
        axes[1+ie//2, ie%2].plot( X, Y, linestyle=Styles[ia], linewidth=1.0, color='black', label=fr"$\alpha = {alpha}$" )
        
    axes[1+ie//2, ie%2].set_xlim(-100, 20)
    axes[1+ie//2, ie%2].set_ylim(-60, 60)
    axes[1+ie//2, ie%2].set_aspect(0.8)
    
    if (abs(e) < 0.01): axes[1+ie//2, ie%2].set_title( rf"$e = {e:.2f}$ (Shue97)" )
    else: axes[1+ie//2, ie%2].set_title( rf"$e = {e:.2f}$" )
    
    axes[1+ie//2, ie%2].scatter( [0], [0], s=1.0, c="black" )

# fig.suptitle( r"$r = r_{0} \cdot \dfrac{1+e}{1+e \cdot \cos \theta} \left(  \dfrac{2}{1 + \cos \theta}\right)^{\alpha}$ for $\alpha$ going from " + f"${np.min(Alpha):.2f}$ (blue) to ${np.max(Alpha):.2f}$ (pink)" )

subfig = fig.add_subfigure(gridspec[0, :])
ax_legend = subfig.add_subplot(1, 1, 1)

h, l = axes[1,1].get_legend_handles_labels()
ax_legend.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -13), ncols=4)
ax_legend.axis("off")

plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

plt.savefig("../images/shue97.svg")
