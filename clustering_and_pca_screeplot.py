import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler ### scaling
from sklearn.cluster import KMeans, DBSCAN  ### k-means and DBscan
from scipy.cluster.hierarchy import linkage, dendrogram ### hierarchical clustering
import seaborn as sns
import matplotlib.pyplot as plt

### For PCA and 3D graphs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Prep data
data_file = open("/home/saran/main/code/MLKTH/exercises/Project/data/data_kaggle_noletters_pruned.csv", "r")
df = pd.read_csv(data_file, sep=",", index_col=["id"])
data_file.close()

# Separate features and target
X = df.drop('diagnosis', axis=1)  # Features (all columns except diagnosis)
y = df['diagnosis']  # Target variable (diagnosis)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)      
data_scaled = pd.DataFrame(data_scaled, columns=X.columns)
print("Data standardized.") 

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
df['KMeans_Labels'] = kmeans.labels_
print("K-Means clustering completed.")

# K-Means cluster centers and inertia
print(f"K-Means inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

# Elbow method to find optimal number of clusters
def find_optimal_clusters(data, max_k=10):
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(data)
        inertias.append(kmeans_temp.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('kmeans_elbow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return k_range, inertias

print("\nFinding optimal number of clusters using elbow method...")
k_range, inertias = find_optimal_clusters(data_scaled)      
print("Elbow method plot saved as 'kmeans_elbow.png'.")

##PCA and 2D Visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca['KMeans_Labels'] = df['KMeans_Labels'] 
print("PCA transformation to 2D completed.")

# Comprehensive PCA Analysis and Visualization
print("\n=== PCA Analysis ===")

# Perform PCA with all components for analysis
pca_full = PCA()
pca_full.fit(data_scaled)

# Plot 1: Explained Variance Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Explained variance ratio
axes[0, 0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
               pca_full.explained_variance_ratio_)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Explained Variance by Component')
axes[0, 0].grid(True, alpha=0.3)

# Cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80% variance')
axes[0, 1].axhline(y=0.9, color='orange', linestyle='--', label='90% variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# First two principal components scatter
cluster_colors = df['KMeans_Labels'].values
scatter = axes[1, 0].scatter(data_pca['PC1'], data_pca['PC2'], 
                            c=cluster_colors, cmap='viridis', alpha=0.7)
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
axes[1, 0].set_title('First Two Principal Components (K-Means Colored)')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='K-Means Cluster')

# Feature loadings for first two components
loadings = pca.components_.T
feature_names = data_scaled.columns  

axes[1, 1].scatter(loadings, loadings, alpha=0.7)
for i, feature in enumerate(feature_names):
    axes[1, 1].annotate(feature, (loadings[i, 0], loadings[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('PC1 Loading')
axes[1, 1].set_ylabel('PC2 Loading')
axes[1, 1].set_title('Feature Loadings')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print PCA statistics
print(f"Total features: {len(pca_full.explained_variance_ratio_)}")
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total variance explained by first 2 PCs: {sum(pca.explained_variance_ratio_):.2%}")

# Find number of components needed for different variance thresholds
for threshold in [0.8, 0.9, 0.95]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"Components needed for {threshold*100:.0f}% variance: {n_components}")

# Detailed PCA biplot
plt.figure(figsize=(12, 8))

# Scatter plot of data points
cluster_colors = df['KMeans_Labels'].values
scatter = plt.scatter(data_pca['PC1'], data_pca['PC2'], 
                     c=cluster_colors, cmap='viridis', alpha=0.6, s=50)

# Add loading vectors (arrows) for top contributing features
loadings = pca.components_.T
loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
top_features_idx = np.argsort(loading_magnitudes)[-8:]  # Top 8 features

scale_factor = 3  # Scale factor for visibility
for i in top_features_idx:
    plt.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor, 
              head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    plt.text(loadings[i, 0] * scale_factor * 1.1, loadings[i, 1] * scale_factor * 1.1, 
             data_scaled.columns[i], fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Biplot: Data Points and Feature Loadings')
plt.colorbar(scatter, label='K-Means Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("PCA analysis plots created and saved.")
print("Files saved: 'pca_analysis.png', 'pca_biplot.png'")

# Scree Plot Analysis
print("\n=== Scree Plot Analysis ===")

# Create comprehensive scree plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Classic Scree Plot - Eigenvalues
eigenvalues = pca_full.explained_variance_
axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Scree Plot - Eigenvalues')
axes[0].grid(True, alpha=0.3)

# Add annotations for first few components
for i in range(min(5, len(eigenvalues))):
    axes[0].annotate(f'{eigenvalues[i]:.2f}', 
                    (i+1, eigenvalues[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')

# Highlight potential "elbow" points
if len(eigenvalues) > 3:
    # Simple elbow detection using second derivative
    second_derivative = np.diff(eigenvalues, 2)
    elbow_point = np.argmax(second_derivative) + 2  # +2 because of double diff
    axes[0].axvline(x=elbow_point+1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Potential Elbow (PC{elbow_point+1})')
    axes[0].legend()

# Scree Plot - Explained Variance Ratio
axes[1].plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
            pca_full.explained_variance_ratio_, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Scree Plot - Explained Variance Ratio')
axes[1].grid(True, alpha=0.3)

# Add annotations for first few components
for i in range(min(5, len(pca_full.explained_variance_ratio_))):
    axes[1].annotate(f'{pca_full.explained_variance_ratio_[i]:.3f}', 
                    (i+1, pca_full.explained_variance_ratio_[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')

# Kaiser criterion line (eigenvalue = 1)
if len(eigenvalues) > 0:
    kaiser_components = np.sum(eigenvalues >= 1.0)
    axes[1].axhline(y=1.0/len(eigenvalues), color='green', linestyle='--', alpha=0.7,
                   label=f'Average variance per component ({1.0/len(eigenvalues):.3f})')
    axes[1].legend()

# Cumulative Scree Plot with decision points
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
axes[2].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', 
            linewidth=2, markersize=8)
axes[2].axhline(y=0.8, color='orange', linestyle='--', alpha=0.8, label='80% variance')
axes[2].axhline(y=0.9, color='red', linestyle='--', alpha=0.8, label='90% variance')
axes[2].axhline(y=0.95, color='purple', linestyle='--', alpha=0.8, label='95% variance')

# Mark the points where thresholds are crossed
for threshold, color in [(0.8, 'orange'), (0.9, 'red'), (0.95, 'purple')]:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    axes[2].plot(n_comp, threshold, 'o', color=color, markersize=12, alpha=0.7)
    axes[2].annotate(f'PC{n_comp}', (n_comp, threshold), 
                    textcoords="offset points", xytext=(10,0), ha='left')

axes[2].set_xlabel('Number of Components')
axes[2].set_ylabel('Cumulative Explained Variance')
axes[2].set_title('Cumulative Variance - Decision Thresholds')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scree_plot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Scree Plot Interpretation and Recommendations
print("\n=== Scree Plot Analysis Results ===")
print(f"Total number of components: {len(eigenvalues)}")
print(f"Components with eigenvalue > 1 (Kaiser criterion): {np.sum(eigenvalues >= 1.0)}")

# Calculate percentage drops between consecutive components
variance_drops = []
for i in range(len(pca_full.explained_variance_ratio_) - 1):
    drop = pca_full.explained_variance_ratio_[i] - pca_full.explained_variance_ratio_[i+1]
    variance_drops.append(drop)

print(f"\nLargest drops in explained variance:")
drop_indices = np.argsort(variance_drops)[::-1][:3]  # Top 3 drops
for i, idx in enumerate(drop_indices):
    print(f"  {i+1}. Between PC{idx+1} and PC{idx+2}: {variance_drops[idx]:.4f}")

# Recommendations based on different criteria
print(f"\n=== Component Selection Recommendations ===")
print(f"Kaiser Criterion (eigenvalue ≥ 1): {np.sum(eigenvalues >= 1.0)} components")
print(f"80% variance threshold: {np.argmax(cumulative_variance >= 0.8) + 1} components")
print(f"90% variance threshold: {np.argmax(cumulative_variance >= 0.9) + 1} components") 
print(f"95% variance threshold: {np.argmax(cumulative_variance >= 0.95) + 1} components")

if len(eigenvalues) > 3:
    print(f"Elbow method suggestion: ~{elbow_point+1} components")

print(f"\nScree plot saved as 'scree_plot_analysis.png'")

# Enhanced Scree Plot with Eigenvector Analysis
print("\n=== Enhanced Scree Plot with Eigenvectors ===")

# Create a comprehensive figure with eigenvector analysis
fig = plt.figure(figsize=(20, 12))

# Create a grid layout for multiple subplots
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])

# Main scree plot (spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=3, markersize=10)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Eigenvalue', fontsize=12)
ax1.set_title('Scree Plot with Eigenvector Analysis', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Highlight first few components
for i in range(min(6, len(eigenvalues))):
    ax1.annotate(f'PC{i+1}\n{eigenvalues[i]:.2f}', 
                (i+1, eigenvalues[i]), 
                textcoords="offset points", 
                xytext=(0,15), 
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Variance ratio plot (spans 2 columns)
ax2 = fig.add_subplot(gs[0, 2:])
bars = ax2.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
               pca_full.explained_variance_ratio_, 
               color='skyblue', alpha=0.7, edgecolor='navy')
ax2.set_xlabel('Principal Component', fontsize=12)
ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
ax2.set_title('Variance Contribution by Component', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add percentage labels on bars
for i, bar in enumerate(bars[:8]):  # Label first 8 bars
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.1%}', ha='center', va='bottom', fontsize=10)

# Eigenvector heatmaps for first 4 PCs
components_to_show = min(4, len(pca_full.components_))
feature_names = data_scaled.columns

for i in range(components_to_show):
    ax = fig.add_subplot(gs[1 + i//2, i%2])
    
    # Get the eigenvector (loadings) for this component
    eigenvector = pca_full.components_[i]
    
    # Create heatmap of feature contributions
    im = ax.imshow(eigenvector.reshape(-1, 1), cmap='RdBu_r', aspect='auto')
    
    # Set labels
    ax.set_title(f'PC{i+1} Eigenvector\n({pca_full.explained_variance_ratio_[i]:.1%} variance)', 
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=10)
    ax.set_xticks([])
    
    # Show feature names for every few features to avoid overcrowding
    step = max(1, len(feature_names) // 10)
    feature_indices = range(0, len(feature_names), step)
    ax.set_yticks(feature_indices)
    ax.set_yticklabels([feature_names[j] for j in feature_indices], fontsize=8)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Annotate top contributing features
    abs_loadings = np.abs(eigenvector)
    top_features_idx = np.argsort(abs_loadings)[-3:]  # Top 3 features
    
    for idx in top_features_idx:
        ax.annotate(f'{feature_names[idx]}\n{eigenvector[idx]:.3f}', 
                   xy=(0, idx), xytext=(1.2, idx),
                   fontsize=8, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

# Summary plot showing feature importance across components
if components_to_show == 4:
    ax_summary = fig.add_subplot(gs[2, :])
    
    # Create a matrix showing feature loadings across first 4 PCs
    loading_matrix = pca_full.components_[:4].T  # Features x Components
    
    im_summary = ax_summary.imshow(loading_matrix, cmap='RdBu_r', aspect='auto')
    ax_summary.set_title('Feature Loadings Across First 4 Principal Components', 
                        fontsize=14, fontweight='bold')
    ax_summary.set_xlabel('Principal Component', fontsize=12)
    ax_summary.set_ylabel('Features', fontsize=12)
    
    # Set ticks
    ax_summary.set_xticks(range(4))
    ax_summary.set_xticklabels([f'PC{i+1}' for i in range(4)])
    
    # Show every nth feature name to avoid overcrowding
    step = max(1, len(feature_names) // 15)
    feature_indices = range(0, len(feature_names), step)
    ax_summary.set_yticks(feature_indices)
    ax_summary.set_yticklabels([feature_names[j] for j in feature_indices], fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im_summary, ax=ax_summary, fraction=0.046, pad=0.04)
    cbar.set_label('Loading Coefficient', fontsize=10)

plt.tight_layout()
plt.savefig('scree_plot_with_eigenvectors.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed eigenvector analysis
print("\n=== Detailed Eigenvector Analysis ===")

for i in range(min(4, len(pca_full.components_))):
    print(f"\n--- PC{i+1} Analysis ---")
    print(f"Explained Variance: {pca_full.explained_variance_ratio_[i]:.2%}")
    print(f"Eigenvalue: {eigenvalues[i]:.3f}")
    
    eigenvector = pca_full.components_[i]
    abs_loadings = np.abs(eigenvector)
    
    # Top positive contributors
    positive_idx = np.where(eigenvector > 0)[0]
    if len(positive_idx) > 0:
        top_positive = positive_idx[np.argsort(eigenvector[positive_idx])[-3:]]
        print("Top positive contributors:")
        for idx in reversed(top_positive):
            print(f"  {feature_names[idx]}: {eigenvector[idx]:.3f}")
    
    # Top negative contributors
    negative_idx = np.where(eigenvector < 0)[0]
    if len(negative_idx) > 0:
        top_negative = negative_idx[np.argsort(eigenvector[negative_idx])[:3]]
        print("Top negative contributors:")
        for idx in top_negative:
            print(f"  {feature_names[idx]}: {eigenvector[idx]:.3f}")
    
    # Overall top contributors by absolute value
    top_overall = np.argsort(abs_loadings)[-5:]
    print("Top contributors by absolute magnitude:")
    for idx in reversed(top_overall):
        print(f"  {feature_names[idx]}: {eigenvector[idx]:.3f}")

print(f"\nEnhanced scree plot with eigenvectors saved as 'scree_plot_with_eigenvectors.png'")

# 2D Scree Plot with Eigenvectors
print("\n=== 2D Scree Plot with Eigenvector Visualization ===")

# Create a comprehensive 2D visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Main scree plot
axes[0, 0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=3, markersize=8)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Eigenvalue')
axes[0, 0].set_title('Scree Plot - Eigenvalues')
axes[0, 0].grid(True, alpha=0.3)

# Add elbow indicator
if len(eigenvalues) > 3:
    second_derivative = np.diff(eigenvalues, 2)
    elbow_point = np.argmax(second_derivative) + 2
    axes[0, 0].axvline(x=elbow_point+1, color='red', linestyle='--', alpha=0.7, 
                      label=f'Elbow at PC{elbow_point+1}')
    axes[0, 0].legend()

# Variance ratio plot
axes[0, 1].bar(range(1, min(11, len(pca_full.explained_variance_ratio_) + 1)), 
               pca_full.explained_variance_ratio_[:10], 
               color='lightcoral', alpha=0.7)
axes[0, 1].set_xlabel('Principal Component')
axes[0, 1].set_ylabel('Explained Variance Ratio')
axes[0, 1].set_title('Variance Explained (First 10 PCs)')
axes[0, 1].grid(True, alpha=0.3)

# Cumulative variance
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
axes[0, 2].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'go-', linewidth=2)
axes[0, 2].axhline(y=0.8, color='orange', linestyle='--', label='80%')
axes[0, 2].axhline(y=0.9, color='red', linestyle='--', label='90%')
axes[0, 2].set_xlabel('Number of Components')
axes[0, 2].set_ylabel('Cumulative Variance')
axes[0, 2].set_title('Cumulative Explained Variance')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 2D Eigenvector plots for PC combinations
pc_combinations = [(0, 1), (0, 2), (1, 2)]  # PC1-PC2, PC1-PC3, PC2-PC3
titles = ['PC1 vs PC2 Eigenvectors', 'PC1 vs PC3 Eigenvectors', 'PC2 vs PC3 Eigenvectors']

for idx, (pc1, pc2) in enumerate(pc_combinations):
    ax = axes[1, idx]
    
    # Get eigenvectors for the two PCs
    eigenvec1 = pca_full.components_[pc1]
    eigenvec2 = pca_full.components_[pc2]
    
    # Create 2D plot with feature vectors
    ax.scatter(eigenvec1, eigenvec2, alpha=0.7, s=60, c='blue', edgecolors='black')
    
    # Add feature labels for top contributors
    feature_magnitudes = np.sqrt(eigenvec1**2 + eigenvec2**2)
    top_features_idx = np.argsort(feature_magnitudes)[-8:]  # Top 8 features
    
    for i in top_features_idx:
        ax.annotate(feature_names[i], 
                   (eigenvec1[i], eigenvec2[i]), 
                   xytext=(5, 5), 
                   textcoords='offset points', 
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # Add arrows from origin to feature points for top contributors
    for i in top_features_idx:
        ax.arrow(0, 0, eigenvec1[i], eigenvec2[i], 
                head_width=0.01, head_length=0.01, 
                fc='red', ec='red', alpha=0.6, linewidth=1.5)
    
    # Add grid lines at zero
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel(f'PC{pc1+1} Loadings ({pca_full.explained_variance_ratio_[pc1]:.1%})')
    ax.set_ylabel(f'PC{pc2+1} Loadings ({pca_full.explained_variance_ratio_[pc2]:.1%})')
    ax.set_title(titles[idx])
    ax.grid(True, alpha=0.3)
    
    # Add quadrant information
    ax.text(0.05, 0.95, f'Q1: +PC{pc1+1}/+PC{pc2+1}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=8)
    ax.text(0.05, 0.05, f'Q3: -PC{pc1+1}/-PC{pc2+1}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=8)

plt.tight_layout()
plt.savefig('scree_plot_2d_eigenvectors.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional 2D circular eigenvector plot (unit circle representation)
print("\n=== Circular Eigenvector Representation ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# PC1 vs PC2 circular plot
ax1 = axes[0]
eigenvec1 = pca_full.components_[0]
eigenvec2 = pca_full.components_[1]

# Create unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')

# Plot feature vectors
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
for i, (name, color) in enumerate(zip(feature_names, colors)):
    ax1.arrow(0, 0, eigenvec1[i], eigenvec2[i], 
             head_width=0.03, head_length=0.03, 
             fc=color, ec=color, alpha=0.7, linewidth=2)
    
    # Add labels for vectors outside certain radius
    magnitude = np.sqrt(eigenvec1[i]**2 + eigenvec2[i]**2)
    if magnitude > 0.15:  # Only label significant vectors
        ax1.text(eigenvec1[i]*1.1, eigenvec2[i]*1.1, name, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax1.set_xlim(-0.6, 0.6)
ax1.set_ylim(-0.6, 0.6)
ax1.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%})')
ax1.set_title('PC1 vs PC2 - Circular Eigenvector Plot')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# PC1 vs PC3 circular plot
ax2 = axes[1]
eigenvec3 = pca_full.components_[2]

ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')

for i, (name, color) in enumerate(zip(feature_names, colors)):
    ax2.arrow(0, 0, eigenvec1[i], eigenvec3[i], 
             head_width=0.03, head_length=0.03, 
             fc=color, ec=color, alpha=0.7, linewidth=2)
    
    magnitude = np.sqrt(eigenvec1[i]**2 + eigenvec3[i]**2)
    if magnitude > 0.15:
        ax2.text(eigenvec1[i]*1.1, eigenvec3[i]*1.1, name, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax2.set_xlim(-0.6, 0.6)
ax2.set_ylim(-0.6, 0.6)
ax2.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC3 ({pca_full.explained_variance_ratio_[2]:.1%})')
ax2.set_title('PC1 vs PC3 - Circular Eigenvector Plot')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('circular_eigenvector_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature clustering based on eigenvector similarity
print("\n=== Feature Grouping by Eigenvector Patterns ===")

# Calculate feature similarities in PC space
pc_loadings = pca_full.components_[:4].T  # First 4 PCs, transposed to features x PCs

# Perform hierarchical clustering on features based on their PC loadings
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Calculate distances between features based on their PC loadings
feature_distances = pdist(pc_loadings, metric='euclidean')
feature_linkage = linkage(feature_distances, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(feature_linkage, labels=feature_names, orientation='top', 
           leaf_rotation=45, leaf_font_size=10)
plt.title('Feature Clustering Based on PC Loading Patterns')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('feature_clustering_eigenvectors.png', dpi=300, bbox_inches='tight')
plt.show()

print("2D eigenvector visualizations saved:")
print("- 'scree_plot_2d_eigenvectors.png'")
print("- 'circular_eigenvector_plot.png'") 
print("- 'feature_clustering_eigenvectors.png'")

