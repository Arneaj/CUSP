from __future__ import annotations
from . import _mag_cusps as _mc

import numpy as np
import joblib
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from typing import Self
from sklearn.preprocessing import StandardScaler

#########################################
# C++ wheels 
#########################################

def preprocess(
    mat: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    new_shape: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Transform a matrix from a non uniform grid to a uniform grid, of shape `new_shape`,
    using X, Y and Z containing the actual position of the center of each grid of each matrix indices. 

    Parameters
    ----------
    mat : np.ndarray
        Input matrix array of doubles, shape (x, y, z, i).
    X, Y, Z : np.ndarray
        Input matrices used in orthonormalisation.
    new_shape : np.ndarray
        Integer array of shape (4,) specifying new shape dimensions.

    Returns
    -------
    np.ndarray
        Orthonormalised matrix as a NumPy array with shape matching `new_shape`.
    """
    return _mc.preprocess(mat, X, Y, Z, new_shape)



def get_bowshock_radius(
    theta: float,
    phi: float,
    Rho: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    dr: float
) -> float:
    """
    Calculate bowshock radius given angles and input data.

    Parameters
    ----------
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.
    Rho : np.ndarray
        Density matrix array.
    earth_pos : np.ndarray
        Earth position vector of shape (3,).
    dr : float
        Step size for radius calculation.

    Returns
    -------
    float
        Computed bowshock radius.
    """
    return _mc.get_bowshock_radius(theta, phi, Rho, earth_pos, dr)

def get_bowshock(
    Rho: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    dr: float,
    nb_phi: int,
    max_nb_theta: int
) -> NDArray[np.float64]:
    """
    Find the bow shock by finding the radius at which dRho_dr * r**3 is minimum,
    casting rays from the earth_pos at angles (theta, phi)

    Parameters
    ----------
    Rho : np.ndarray
        Density matrix array of shape (x,y,z,).
    earth_pos : np.ndarray
        Earth position vector of shape (3,).
    dr : float
        Step size for radius calculation.
    nb_phi : int
        Number of divisions in phi.
    max_nb_theta : int
        Maximum number of divisions in theta.

    Returns
    -------
    np.ndarray
        Array of points with shape (N, 3) representing bowshock coordinates.
    """
    return _mc.get_bowshock(Rho, earth_pos, dr, nb_phi, max_nb_theta)


def get_interest_points(
    J_norm: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    Rho: NDArray[np.float64],
    theta_min: float,
    theta_max: float,
    nb_theta: int,
    nb_phi: int,
    dx: float,
    dr: float,
    alpha_0_min: float,
    alpha_0_max: float,
    nb_alpha_0: int,
    r_0_mult_min: float,
    r_0_mult_max: float,
    nb_r_0: int,
    avg_std_dev: float | None = None
) -> NDArray[np.float64]:
    """
    Calculate interest points from inputs.

    Parameters
    ----------
    J_norm : np.ndarray
        Normalized current density matrix of shape (x,y,z,i,).
    earth_pos : np.ndarray
        Earth position vector of shape (3,).
    Rho : np.ndarray
        Density matrix array of shape (x,y,z,).
    theta_min, theta_max : float
        Angle bounds for theta.
    nb_theta, nb_phi : int
        Number of divisions for theta and phi.
    dx, dr : float
        Step sizes.
    alpha_0_min, alpha_0_max : float
        Bounds for alpha_0.
    nb_alpha_0 : int
        Number of alpha_0 divisions.
    r_0_mult_min, r_0_mult_max : float
        Multiplicative range for r_0 where r_0 = r_0_mult * r_I with r_I the inner radius in the simulation.
    nb_r_0 : int
        Number of r_0 divisions.
    avg_std_dev : Optional[float]
        Optional output parameter for average standard deviation.

    Returns
    -------
    np.ndarray
        Interest points array with shape (nb_theta*nb_phi, 4).
    """
    return _mc.get_interest_points(J_norm, earth_pos, Rho, 
                                   theta_min, theta_max, nb_theta, nb_phi,
                                   dx, dr,
                                   alpha_0_min, alpha_0_max, nb_alpha_0,
                                   r_0_mult_min, r_0_mult_max, nb_r_0,
                                   avg_std_dev)

def process_interest_points(
    interest_points: NDArray[np.float64],
    nb_theta: int,
    nb_phi: int,
    shape_sim: NDArray[np.int32],
    shape_real: NDArray[np.int32],
    earth_pos_sim: NDArray[np.float64],
    earth_pos_real: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform interest points from simulation coordinates to real coordinates.

    Parameters
    ----------
    interest_points : np.ndarray
        Interest points array with shape (N, 4).
    nb_theta, nb_phi : int
        Number of divisions in theta and phi.
    shape_sim, shape_real : np.ndarray
        Shape arrays of shape (4,) describing simulation and real data shapes.
    earth_pos_sim, earth_pos_real : np.ndarray
        Earth position vectors of shape (3,) for simulation and real.

    Returns
    -------
    np.ndarray
        Processed interest points array of shape (N, 4).
    """
    return _mc.process_interest_points(interest_points, nb_theta, nb_phi, shape_sim, shape_real, earth_pos_sim, earth_pos_real)

def process_points(
    points: NDArray[np.float64],
    shape_sim: NDArray[np.int32],
    shape_real: NDArray[np.int32],
    earth_pos_sim: NDArray[np.float64],
    earth_pos_real: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform points from simulation coordinates to real coordinates.

    Parameters
    ----------
    points : np.ndarray
        Points array with shape (N, 3).
    shape_sim, shape_real : np.ndarray
        Shape arrays of shape (4,) describing simulation and real data shapes.
    earth_pos_sim, earth_pos_real : np.ndarray
        Earth position vectors of shape (3,) for simulation and real.

    Returns
    -------
    np.ndarray
        Processed points array of shape (N, 3).
    """
    return _mc.process_points(points, shape_sim, shape_real, earth_pos_sim, earth_pos_real)


def Shue97(
    params: NDArray[np.float64], 
    theta: float | NDArray[np.float64]
) -> float | NDArray[np.float64] | None:
    """
    Analytical approximation of the Magnetopause topology as written by Shue in his 1997 paper.

    Parameters
    ----------
    params : np.ndarray
        Parameters array with shape (2,).
    theta : float or np.ndarray
        Angle at which the radius should be calculated. 
    
    Returns
    -------
    float or np.ndarray
        Radius at this angle.
    """
    if params.size != 2:
        print("ERROR: there should be 2 parameters for the Shue97 function")
        return None
    return _mc.Shue97(params, theta)

def Liu12(
    params: NDArray[np.float64], 
    theta: float | NDArray[np.float64], 
    phi: float | NDArray[np.float64]
) -> float | NDArray[np.float64] | None:
    """
    Analytical approximation of the Magnetopause topology as written by Liu in his 2012 paper.

    Parameters
    ----------
    params : np.ndarray
        Parameters array with shape (10,).
    theta, phi : float or np.ndarray
        Angle at which the radius should be calculated. 
    
    Returns
    -------
    float or np.ndarray
        Radius at this angle.
    """
    if params.size != 10:
        print("ERROR: there should be 10 parameters for the Liu12 function")
        return None
    return _mc.Liu12(params, theta, phi) 
    
def Rolland25(
    params: NDArray[np.float64], 
    theta: float | NDArray[np.float64], 
    phi: float | NDArray[np.float64] 
) -> float | NDArray[np.float64] | None:
    """
    Analytical approximation of the Magnetopause topology as written by Rolland in his 2025 thesis.

    Parameters
    ----------
    params : np.ndarray
        Parameters array with shape (11,).
    theta, phi : float or np.ndarray
        Angle at which the radius should be calculated. 
    
    Returns
    -------
    float or np.ndarray
        Radius at this angle.
    """
    if params.size != 11:
        print("ERROR: there should be 11 parameters for the Rolland25 function")
        return None
    return _mc.Rolland25(params, theta, phi)


def fit_to_analytical(
    interest_points: NDArray[np.float64],
    initial_params: NDArray[np.float64],
    lowerbound: NDArray[np.float64],
    upperbound: NDArray[np.float64],
    radii_of_variation: NDArray[np.float64],
    analytical_function: str = "Rolland25",
    nb_runs: int = 10,
    max_nb_iterations_per_run: int = 50,
) -> tuple[NDArray[np.float64], float] | None:
    """
    Analytical fitting of the Shue97, Liu12 or Rolland25 analytical functions 
    to an array of interest points. N equals respectively 2, 10 and 11.

    Parameters
    ----------
    interest_points : np.ndarray
        Interest point array to fit to of shape (`nb_interest_points`, 4).
    initial_parameters : np.ndarray
        Parameters array with shape (N,).
    lowerbound, upperbound : np.ndarray
        Parameters array with shape (N,) corresponding to the lower and upper bounds
        that the parameters can take during fitting.
    radii_of_variation : np.ndarray
        Parameters array with shape (N,) corresponding to the maximum distance each 
        of the parameters will randomly move away for the initial_params at the 
        beginning of a run.
    nb_runs : int
        Number of times the fitting algorithm will start again with other randomly 
        selected initial parameters.
    max_nb_iterations_per_run : int
        Maximum number of iterations the fitting algorithm will do before stopping
        even if it hasn't converged.
    
    Returns
    -------
    (np.ndarray, float)
        Array of the final parameters after fit and the fitting cost of these parameters. 
    """
    if analytical_function == "Shue97":
        if initial_params.size != 2:
            print("ERROR: there should be 2 parameters for the Shue97 function")
            return None
        return _mc.fit_to_Shue97(interest_points, initial_params, lowerbound, upperbound, 
                                 radii_of_variation, nb_runs, max_nb_iterations_per_run)
    if analytical_function == "Liu12":
        if initial_params.size != 10:
            print("ERROR: there should be 10 parameters for the Liu12 function")
            return None
        return _mc.fit_to_Liu12(interest_points, initial_params, lowerbound, upperbound, 
                                radii_of_variation, nb_runs, max_nb_iterations_per_run)
    if analytical_function == "Rolland25":
        if initial_params.size != 11:
            print("ERROR: there should be 11 parameters for the Rolland25 function")
            return None
        return _mc.fit_to_Rolland25(interest_points, initial_params, lowerbound, upperbound, 
                                    radii_of_variation, nb_runs, max_nb_iterations_per_run)
    print("ERROR: analytical function should be 'Shue97', 'Liu12' or 'Rolland25'")
    return None


def get_grad_J_fit_over_ip(
    params: NDArray[np.float64],
    interest_points: NDArray[np.float64],
    J_norm: NDArray[np.float64], earth_pos: NDArray[np.float64],
    analytical_function: str = "Rolland25",
    dx: float = 0.5, dy: float = 0.5, dz: float = 0.5
) -> float | None:
    """
    Ratio of the current density gradient along the magnetopause between the 
    Shue97, Liu12 or Rolland25 analytical functions and the interest points. 
    N equals respectively 2, 10 and 11.
    
    Parameters
    ----------
    params : np.ndarray
        Parameters for the Shue97 function of shape (N,).
    interest_points : np.ndarray
        Interest point array of shape (`nb_interest_points`, 4).
    J_norm : np.ndarray
        Normalised current density matrix of shape (X, Y, Z).
    earth_pos : np.ndarray
        Position of the Earth of shape (3,).
    dx, dy, dz : Optional[int]
        Used to calculate the gradient. Default value is 0.5.
    
    Returns
    -------
    float
        ||grad(||J_fit||)|| / ||grad(||J_ip||)||.
    """
    if analytical_function == "Shue97":
        if params.size != 2:
            print("ERROR: there should be 2 parameters for the Shue97 function")
            return None
        return _mc.get_grad_J_fit_over_ip_Shue97(params, interest_points, J_norm, earth_pos, dx, dy, dz)
    if analytical_function == "Liu12":
        if params.size != 10:
            print("ERROR: there should be 10 parameters for the Liu12 function")
            return None
        return _mc.get_grad_J_fit_over_ip_Liu12(params, interest_points, J_norm, earth_pos, dx, dy, dz)
    if analytical_function == "Rolland25":
        if params.size != 11:
            print("ERROR: there should be 11 parameters for the Rolland25 function")
            return None
        return _mc.get_grad_J_fit_over_ip_Rolland25(params, interest_points, J_norm, earth_pos, dx, dy, dz)
    print("ERROR: analytical function should be 'Shue97', 'Liu12' or 'Rolland25'")
    return None


def interest_point_flatness_checker(
    interest_points: NDArray[np.float64],
    nb_theta: int, nb_phi: int,
    threshold: float | None = None, phi_radius: float | None = None
) -> tuple[float, bool]:
    """
    Checks in the (earth_pos,x,z) plane at what angle the interest points recede towards +x
    past a given threshold. Will also say if the interest points are concave, as an extreme case
    
    Parameters
    ----------
    interest_points : np.ndarray
        Interest point array of shape (`nb_interest_points`, 4).
    nb_theta, nb_phi : int
        Number of phi and theta.
    threshold : Optional[float]
        How many grid cells before the dayside is considered to have receded.
        Default value is 2.0.
    phi_radius : Optional[float]
        The angle phi to consider both sides of the (earth_pos,x,z) plane to average.
        out any possible outliers in the plane. Default value is 0.3.
    
    Returns
    -------
    (float, bool)
        Returns the angle at which the day-side stops being considered flat, and whether it was concave.
    """
    return _mc.interest_point_flatness_checker(interest_points, nb_theta, nb_phi, threshold, phi_radius)


#########################################
# Python additional features
#########################################


class MagCUSPS_Model:
    def define(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def load(self, path: str) -> Self:
        """
        Load a pickled MagCUSPS_model object 
        """
        self = joblib.load(path)
        return self
        
    def dump(self, path: str):
        """
        Pickle an entire MagCUSPS_model object
        """
        joblib.dump(self, path)
        
    def predict(self, X):
        """
        Scale the data with the self.scaler and predict the output with self.model
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class MagCUSPS_RandomForestModel(MagCUSPS_Model):
    def define(self, model: RandomForestRegressor, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler
    
    def get_sample_uncertainty(self, X_sample):
        """
        Get uncertainty from Random Forest model
        """
        tree_predictions = np.array([tree.predict(X_sample.reshape(1, -1))[0] 
                                    for tree in self.model.estimators_])
        return np.std(tree_predictions) 
    
    def get_batch_uncertainty(self, X):
        """
        Get uncertainty of entire batch from Random Forest model
        """
        uncertainties = []
        for i in range(len(X)):
            sample_unc = self.get_sample_uncertainty(X[i])
            uncertainties.append(sample_unc)
        return np.mean(uncertainties)



def analyse():
    pass

def predict_with_model(input_features, model=None):
    """
    Make predictions using either a provided model or a saved model
    
    Parameters
    ----------
        input_features: np.ndarray
            Input data for prediction. If using the default model, 
            input should be the output of the provided `analyse` function.
        model: sklearn model object, optional
            User-pretrained model used for the prediction in the place of the
            model pretrained from Gorgon data.
    
    Returns
    -------
        predictions: 
            array with values between 0 and 1 of shape (N,)
    """
    
    model_path='default_evaluation_prediction_model.pkl'
    
    # Use provided model or load from file
    if model is None:
        model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(input_features)
    
    # Ensure predictions are between 0 and 1
    predictions = np.clip(predictions, 0, 1)
    
    return predictions



__all__ = [
    "preprocess",
    "get_bowshock_radius",
    "get_bowshock",
    "get_interest_points",
    "process_interest_points",
    "process_points",
    "Shue97",
    "Liu12",
    "Rolland25",
    "fit_to_analytical",
    "get_grad_J_fit_over_ip",
    "interest_point_flatness_checker",
    "analyse",
    "predict_with_model",
]
