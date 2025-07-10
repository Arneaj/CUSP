import numpy as np
import matplotlib.pyplot as plt
from gorgon import import_from



def get_earth_pos( B_norm: np.ndarray ) -> tuple[float]:
    return np.unravel_index(np.argmax(B_norm, axis=None), B_norm.shape)



def main():
    """## Read file and extract values"""

    B,V,T,rho = import_from("../data/Run1_28800")
    
    B_norm = np.linalg.norm( B, axis=3 )

    print( B_norm.shape )
    
    max_index = np.unravel_index(np.argmax(B_norm, axis=None), B_norm.shape)
    
    print( max_index )

if __name__ == "__main__":
    main()
