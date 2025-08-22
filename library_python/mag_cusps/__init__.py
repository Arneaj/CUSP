from __future__ import annotations
from .mag_cusps import *

import numpy as np
import joblib

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
