import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


from gorgon import filename, import_from, mu_0




def main():
    B,_,_,_ = import_from(filename)
    
    XMAX = 240
    XMIN = 0

    YMAX = 160
    YMIN = 0

    ZMAX = 160
    ZMIN = 0

    STEP = 2
    
    u = 0.5 * np.linalg.norm(B[XMIN:XMAX:STEP,YMIN:YMAX:STEP,ZMIN:ZMAX:STEP,], axis=3)**2 / mu_0
    
    plot_mask = (u < np.median(u)*5e0)
    
    # NB_LAYERS = 3
    
    # layers = np.linspace( np.min(u), np.max(u), NB_LAYERS )
    
    # u_masks = []
    
    # for layer in layers:
    #     u_masks.append( np.abs(u - layer) < 1e-3 )
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    
    frame = ax.imshow( u[0,:,:], cmap="copper", vmax=1e-10 )
    # ax.imshow( np.ones_like(u[:,0,:]), cmap="Greys", alpha=np.array(plot_mask[:,0,:], dtype=np.float32) )
    
    def animate(i):
        fig.suptitle(f"y index = {i}")
        
        ax.clear()
        frame = ax.imshow( u[i,:,:], cmap="copper", vmax=1e-10 )
        # ax.imshow( np.ones_like(u[:,i,:]), cmap="Greys", alpha=np.array(plot_mask[:,i,:], dtype=np.float32) )
        
        if i % 10 == 0:
            print(f"frame {i} done")
        
        return frame
        
    ani = anim.FuncAnimation(fig, animate, interval=50, frames=np.arange(0, 160//STEP))  
    ani.save(filename="potential.gif", writer="pillow")
    
    
    
    
if __name__=="__main__":
    main()