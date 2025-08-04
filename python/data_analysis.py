import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


db = pd.read_csv( "../result_data.csv" )
db.replace( "Perfect", 1.0, inplace=True )
db.replace( "Ok", 0.66, inplace=True )
db.replace( "Eh", 0.33, inplace=True )
db.replace( "Bad", 0.0, inplace=True )

db.drop(columns=db.columns[0:4], axis=1, inplace=True)
db.drop("V_y", axis=1, inplace=True)
db.drop("V_z", axis=1, inplace=True)
db.drop("B_x", axis=1, inplace=True)
db.drop("B_y", axis=1, inplace=True)

data = db.to_numpy( np.float64 )
names = db.columns

inputs = data[:,3:]
outputs = data[:,2]

input_names = np.array( names[3:] )
output_names = np.array( names[0] )

print( "inputs:", input_names )
print( "outputs:", output_names )

def comprehensive_regression_analysis(X, y, feature_names):
    """
    Comprehensive regression analysis for quality score prediction
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store models and results
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("=== MODEL PERFORMANCE COMPARISON ===")
    print(f"{'Model':<20} {'R² Score':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 50)
    
    # Train and evaluate each model
    for name, model in models.items():
        if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
        
        print(f"{name:<20} {r2:.3f}      {rmse:.3f}      {mae:.3f}")
    
    return results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def analyze_feature_importance(results, feature_names, X_train, X_train_scaled):
    """
    Analyze which features are most important for predicting quality
    """
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Random Forest feature importance
    rf_model = results['Random Forest']['model']
    rf_importance = rf_model.feature_importances_
    
    # Gradient Boosting feature importance  
    gb_model = results['Gradient Boosting']['model']
    gb_importance = gb_model.feature_importances_
    
    # Linear regression coefficients (from scaled data)
    ridge_model = results['Ridge Regression']['model']
    ridge_coefs = np.abs(ridge_model.coef_)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Random_Forest': rf_importance,
        'Gradient_Boosting': gb_importance,
        'Ridge_Coeffs': ridge_coefs / ridge_coefs.max()  # Normalize
    })
    
    # Sort by average importance
    importance_df['Average'] = (importance_df['Random_Forest'] + 
                               importance_df['Gradient_Boosting'] + 
                               importance_df['Ridge_Coeffs']) / 3
    importance_df = importance_df.sort_values('Average', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10)[['Feature', 'Average']].to_string(index=False))
    
    return importance_df

def create_visualizations(results, y_test, importance_df, X, y):
    """
    Create comprehensive visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quality Score Prediction Analysis', fontsize=16)
    
    # 1. Model performance comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    axes[0, 0].bar(model_names, r2_scores, color='skyblue')
    axes[0, 0].set_title('Model R² Scores')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE comparison
    axes[0, 1].bar(model_names, rmse_scores, color='lightcoral')
    axes[0, 1].set_title('Model RMSE')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Best model predictions vs actual
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_predictions = results[best_model_name]['predictions']
    
    axes[0, 2].scatter(y_test, best_predictions, alpha=0.7)
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 2].set_xlabel('Actual Quality Score')
    axes[0, 2].set_ylabel('Predicted Quality Score')
    axes[0, 2].set_title(f'Best Model: {best_model_name}')
    
    # 4. Feature importance
    top_features = importance_df.head(10)
    axes[1, 0].barh(top_features['Feature'], top_features['Average'])
    axes[1, 0].set_title('Top 10 Feature Importance')
    axes[1, 0].set_xlabel('Importance Score')
    
    # 5. Quality score distribution
    axes[1, 1].hist(y, bins=20, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Quality Score (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Quality Score Distribution')
    
    # 6. Feature correlation with quality
    # Show correlation of top 5 features with quality score
    top_5_features = importance_df.head(5)['Feature'].tolist()
    feature_indices = [np.argwhere(input_names == f)[0,0] for f in top_5_features]
    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in feature_indices]
    
    axes[1, 2].bar(top_5_features, correlations, color='orange')
    axes[1, 2].set_title('Top 5 Features - Correlation with Quality')
    axes[1, 2].set_ylabel('Correlation Coefficient')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig("../images/data_analysis.svg")

def pca_for_regression(X, y, feature_names):
    """
    Use PCA to see if we can predict quality with fewer dimensions
    """
    print(f"\n=== PCA FOR DIMENSIONALITY REDUCTION ===")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of components
    n_components_list = [2, 3, 5, 8, 10]
    pca_results = {}
    
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        
        # Train model on PCA features
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.3, random_state=42
        )
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_pca, y_train)
        y_pred = rf.predict(X_test_pca)
        
        r2 = r2_score(y_test, y_pred)
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        pca_results[n_comp] = {
            'r2': r2,
            'explained_variance': explained_var,
            'pca': pca
        }
        
        print(f"{n_comp} components: R² = {r2:.3f}, "
              f"Explains {explained_var:.1%} of feature variance")
    
    return pca_results

# Run the complete analysis
print("🔬 PHYSICS SIMULATION QUALITY PREDICTION ANALYSIS")
print("=" * 60)

# Main regression analysis
results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \
    comprehensive_regression_analysis(inputs, outputs, input_names)

# Feature importance analysis
importance_df = analyze_feature_importance(results, input_names, X_train, X_train_scaled)

# Create visualizations
create_visualizations(results, y_test, importance_df, inputs, outputs)

# PCA analysis for dimensionality reduction
pca_results = pca_for_regression(inputs, outputs, input_names)

# Practical recommendations
print(f"\n=== PRACTICAL RECOMMENDATIONS ===")
best_model = max(results.keys(), key=lambda k: results[k]['r2'])
best_r2 = results[best_model]['r2']

print(f"✅ Best model: {best_model} (R² = {best_r2:.3f})")
print(f"✅ Top 3 most important features:")
for i, row in importance_df.head(3).iterrows():
    print(f"   {i+1}. {row['Feature']} (importance: {row['Average']:.3f})")

print(f"✅ You could potentially use just 5 PCA components and still get")
print(f"   R² = {pca_results[5]['r2']:.3f} while explaining")
print(f"   {pca_results[5]['explained_variance']:.1%} of the feature variance")

print(f"\n💡 NEXT STEPS:")
print(f"   • Focus on improving the top {len(importance_df[importance_df['Average'] > 0.1])} most important features")
print(f"   • Consider feature engineering (interactions between top features)")
print(f"   • Collect more data if R² < 0.8 for your application needs")