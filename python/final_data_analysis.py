from pyclbr import Class
from random import random
from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

import joblib

def get_rf_uncertainty(model, X_sample):
    """
    Get uncertainty from Random Forest
    """
    if not hasattr(model, 'estimators_'):
        raise ValueError("Model must be RandomForestRegressor")
    
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X_sample.reshape(1, -1))[0] 
                                for tree in model.estimators_])
    
    uncertainty = np.std(tree_predictions)  # Standard deviation as uncertainty
    
    return uncertainty

def get_batch_rf_uncertainty(model, X):
    """
    Get uncertainty from Random Forest
    """
    if not hasattr(model, 'estimators_'):
        raise ValueError("Model must be RandomForestRegressor")
    
    uncertainties = []
    for i in range(len(X)):
        sample_unc = get_rf_uncertainty(model, X[i])
        uncertainties.append(sample_unc)
    return np.mean(uncertainties)

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
                                    for tree in model.estimators_])
        return np.std(tree_predictions) 
    
    def get_batch_uncertainty(self, X):
        """
        Get uncertainty of entire batch from Random Forest model
        """
        uncertainties = []
        for i in range(len(X)):
            sample_unc = get_rf_uncertainty(model, X[i])
            uncertainties.append(sample_unc)
        return np.mean(uncertainties)
    
    

df_inputs = pd.read_csv( "../.result_folder/great_analysis867056.csv", sep="," )
df_labels = pd.read_csv( "../.result_folder/labels.csv", sep="\t" )

combined_df = pd.merge( df_inputs, df_labels, 
                        left_on=['run', 'timestep'], 
                        right_on=['Run_nb', 'Time'], 
                        how='inner')

filtered_df = combined_df#[~combined_df['Model_result'].isin(['Eh', 'Ok'])]

value_map = {'Perfect': 1.0, 'Ok': 0.66, 'Eh': 0.33, 'Bad': 0.0}
filtered_df['Model_result'] = filtered_df['Model_result'].map(value_map)

labels_df = filtered_df.filter(items=['Model_result'])
labels = labels_df.to_numpy( np.float64 ).ravel()


fig, axes = plt.subplots(2, 3)
fig.set_size_inches( 13, 10 )

r2 = np.empty((3))
rmse = np.empty((3))
uncertainty = np.empty((3))

models = ["Shue97", "Liu12", "Rolland25"]


for i, m in enumerate(models):
    selected_columns = [col for col in filtered_df.columns 
                            if (
                            col == 'max_theta_in_threshold' 
                            or col == "is_concave"
                            or m in col
                            )
                            and "_time_taken_s" not in col
                        ]
    inputs_df = filtered_df[selected_columns]
    
    inputs = inputs_df.to_numpy( np.float64 )
    input_names = inputs_df.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.3
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_r2 = 0.7
    best_rmse = None
    best_uncertainty = 0.5
    
    best_model = None
    best_y_pred = None
    
    best_nb_estim = None
    best_max_depth = None
    best_min_spl_split = None
    best_min_spl_leaf = None
    
    best_seed = None
    
    for seed_i in range(20):
        for n_estim in [35, 40, 45, 50, 55, 60, 65, 70]:
            for mx_depth in [2, 4, 6, 8, 10]:
                for min_spl_split in [2, 3, 4, 5, 6, 7]:
                    for min_spl_leaf in [1, 2, 3, 4]:
                        seed = int((random() * 100) // 1)
                        
                        model = RandomForestRegressor(
                            n_estimators=n_estim,       
                            max_depth=mx_depth,           
                            min_samples_split=min_spl_split,    
                            min_samples_leaf=min_spl_leaf,
                            random_state=seed
                        )
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        this_r2 = r2_score(y_test, y_pred)
                        this_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        this_uncertainty = get_batch_rf_uncertainty(model, X_test_scaled)
                        
                        if (this_r2 < 0.75): continue
            
                        if (this_r2/best_r2) * (best_uncertainty/this_uncertainty)**0.5 > 1.0:
                            best_r2 = this_r2
                            best_rmse = this_rmse
                            best_uncertainty = this_uncertainty
                            
                            best_model = model
                            best_y_pred = y_pred
                            
                            best_nb_estim = n_estim
                            best_max_depth = mx_depth
                            best_min_spl_split = min_spl_split
                            best_min_spl_leaf = min_spl_leaf
                            
                            best_seed = seed
        
    if best_model is not None:
        mag_cusps_model = MagCUSPS_RandomForestModel()
        mag_cusps_model.define(best_model, scaler)
    else:
        print(f"ERROR: no {m} model was sufficiently good")
        continue
        
    joblib.dump(mag_cusps_model, f"../.result_folder/evaluation_prediction_model_{m}.pkl")
    
    r2[i] = best_r2
    rmse[i] = best_rmse
    uncertainty[i] = best_uncertainty
    
    print(f"Best {m} model:\n\tR^2={best_r2}\n\tRMSE={best_rmse}\n\tUncertainty={best_uncertainty}")
    print(f"\tNumber of trees={best_nb_estim}\n\tMax depth={best_max_depth}")
    print(f"\tMin sample split={best_min_spl_split}\n\tMin sample leaf={best_min_spl_leaf}")
    print(f"\tSeed={best_seed}")
    
    axes[0, i].scatter(y_test, y_pred, alpha=0.7, color=(0.2, 0.2, 0.2))
    axes[0, i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, i].set_xlabel('Actual Quality Score')
    axes[0, i].set_ylabel('Predicted Quality Score')
    
    axes[0, i].set_title(f"{m}: Actual vs Predicted\nnb_estim={best_nb_estim}, max_depth={best_max_depth}")
    
    
axes[1, 0].bar( models, r2, color=(0.3, 0.3, 0.3), width=0.5 )
axes[1, 0].set_title(f"R^2")

axes[1, 1].bar( models, rmse, color=(0.3, 0.3, 0.3), width=0.5 )
axes[1, 1].set_title(f"RMSE")

axes[1, 2].bar( models, uncertainty, color=(0.3, 0.3, 0.3), width=0.5 )
axes[1, 2].set_title(f"Uncertainty")

    
    
    
plt.savefig("../images/final_data_analysis.svg")
