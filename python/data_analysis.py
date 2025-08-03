import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


db = pd.read_csv( "../result_data.csv" )
db.replace( "Perfect", 1.0, inplace=True )
db.replace( "Ok", 0.66, inplace=True )
db.replace( "Eh", 0.33, inplace=True )
db.replace( "Bad", 0.0, inplace=True )

db.drop(columns=db.columns[0:4], axis=1, inplace=True)

data = db.to_numpy( np.float64 )
names = db.columns

inputs = data[:,3:12]
outputs = np.concatenate( [data[:,:3], data[:,12:]], axis=1 )

input_names = np.array( names[3:12] )
output_names = np.concatenate( [names[:3], names[12:]] )

print( "inputs:", input_names )
print( "outputs:", output_names )

def perform_pca_analysis(data, feature_names, title="PCA Analysis"):
    """
    Perform PCA analysis on the given data
    """
    # Step 1: Standardize the data (important for PCA!)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Step 2: Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Step 3: Analyze results
    print(f"\n=== {title} ===")
    print(f"Original dimensions: {data.shape}")
    print(f"Explained variance ratio by component:")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} "
              f"(cumulative: {cumulative_variance[i]:.3f})")
    
    # Find how many components explain 95% of variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")
    
    # Step 4: Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Results', fontsize=16)
    
    # Plot 1: Scree plot (explained variance)
    axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, 'bo-')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].grid(True)
    
    # Plot 2: Cumulative explained variance
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), 
                   cumulative_variance, 'ro-')
    axes[0, 1].axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Variance Explained')
    axes[0, 1].grid(True)
    
    # Plot 3: First two principal components
    axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1, 0].set_title('Data in PC1-PC2 Space')
    axes[1, 0].grid(True)
    
    # Plot 4: Component loadings for first PC
    if len(feature_names) <= 20:  # Only show if not too many features
        axes[1, 1].bar(range(len(feature_names)), pca.components_[0])
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Loading')
        axes[1, 1].set_title('PC1 Loadings')
        axes[1, 1].set_xticks(range(len(feature_names)))
        axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
    else:
        axes[1, 1].bar(range(len(pca.components_[0])), pca.components_[0])
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Loading')
        axes[1, 1].set_title('PC1 Loadings')
    
    plt.tight_layout()
    fig.savefig( "../images/" + title + ".svg" )
    
    return pca, pca_result, scaler

def interpret_components(pca, feature_names, n_components=3):
    """
    Interpret the first few principal components
    """
    print(f"\n=== Component Interpretation ===")
    
    for i in range(min(n_components, len(pca.components_))):
        print(f"\nPrincipal Component {i+1} "
              f"({pca.explained_variance_ratio_[i]:.1%} of variance):")
        
        # Get component loadings
        loadings = pca.components_[i]
        
        # Find strongest positive and negative loadings
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        
        print("  Strongest contributors:")
        for j in range(min(5, len(loadings))):
            idx = sorted_indices[j]
            sign = "+" if loadings[idx] > 0 else "-"
            print(f"    {sign}{abs(loadings[idx]):.3f} × {feature_names[idx]}")

# Run the analysis
print("Analyzing OUTPUT parameters with PCA...")
pca_outputs, pca_result_outputs, scaler_outputs = perform_pca_analysis(
    outputs, output_names, "Output_Parameters_PCA"
)

interpret_components(pca_outputs, output_names)

# Optional: Also analyze inputs if you want
print("\n" + "="*50)
print("Analyzing INPUT parameters with PCA...")
pca_inputs, pca_result_inputs, scaler_inputs = perform_pca_analysis(
    inputs, input_names, "Input_Parameters_PCA"
)

interpret_components(pca_inputs, input_names)

# Bonus: Show how to use PCA results for further analysis
print(f"\n=== Using PCA Results ===")
print("You can now use the first few principal components for:")
print("1. Clustering: Apply K-means to the first 3-5 PCs")
print("2. Regression: Use PCs as simplified outputs to predict from inputs")
print("3. Visualization: Plot data in PC space to identify patterns")
print(f"4. Dimensionality reduction: {outputs.shape[1]} → {np.sum(pca_outputs.explained_variance_ratio_[:5]):.1%} variance with just 5 components")