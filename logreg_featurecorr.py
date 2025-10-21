
##imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import skewnorm

##Prep data
data_file = open("/home/saran/main/code/MLKTH/exercises/Project/data/data_kaggle_noletters_pruned.csv", "r")
df = pd.read_csv(data_file, sep=",", index_col=["id"])
data_file.close()
#print(df[['diagnosis']])

##Check for N/As
null_matrix = df.isnull()
nollfinns = 0
for i in null_matrix:
    if True in null_matrix[i]:
        nollfinns = 1

##Print statement
print("Are there any 0s? ", nollfinns)

##Replace non-numeric symbols. (did manually)

# Plot the heatmap
plt.rcParams.update({'font.size': 30})
f = plt.figure()
f.set_figwidth(31)
f.set_figheight(31)
correl = df[1:].corr()

print(df.dtypes)

# Get all column names
all_features = df.columns.tolist()
X = df.drop('diagnosis', axis=1)  # Features (all columns except diagnosis)
y = df['diagnosis']  # Target variable (diagnosis)

# Divide features by categories
mean_features = [col for col in all_features if col.endswith('_me')]
se_features = [col for col in all_features if col.endswith('_se')]
worst_features = [col for col in all_features if col.endswith('_wo')]

print(f"Mean features ({len(mean_features)}): {mean_features}")
print(f"Standard Error features ({len(se_features)}): {se_features}")
print(f"Worst features ({len(worst_features)}): {worst_features}")

# Create separate datasets for each category
X_mean = X[mean_features]
X_se = X[se_features]
X_worst = X[worst_features]

# Create enhanced clustermaps for different feature categories

# First, prepare the data by scaling it for better clustering
from sklearn.preprocessing import StandardScaler

# Remove diagnosis column for clustering (keep only numeric features)
X_scaled = StandardScaler().fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Add diagnosis information as color annotation
diagnosis_colors = y.map({0: 'blue', 1: 'red'})  # 0=benign (blue), 1=malignant (red)

print("\nCreating clustermaps...")

# 1. Full dataset clustermap with diagnosis annotation
plt.figure(figsize=(15, 12))
full_clustermap = sns.clustermap(
    X_scaled_df, 
    figsize=(15, 12),
    cmap='viridis',
    standard_scale=None,  
    row_colors=diagnosis_colors,
    col_cluster=True,
    row_cluster=True,
    linewidths=0,
    cbar_kws={'label': 'Standardized Value'},
    dendrogram_ratio=0.1
)
full_clustermap.ax_row_dendrogram.set_title('Sample Clustering', fontsize=12, pad=20)
full_clustermap.ax_col_dendrogram.set_title('Feature Clustering', fontsize=12, pad=20)
plt.suptitle('Full Dataset Clustermap\n(Blue=Benign, Red=Malignant)', y=0.95, fontsize=14)
plt.savefig('full_dataset_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Mean features clustermap
plt.figure(figsize=(12, 10))
X_mean_scaled = StandardScaler().fit_transform(X_mean)
X_mean_scaled_df = pd.DataFrame(X_mean_scaled, columns=X_mean.columns, index=X_mean.index)

mean_clustermap = sns.clustermap(
    X_mean_scaled_df,
    figsize=(12, 10),
    cmap='RdYlBu_r',
    row_colors=diagnosis_colors,
    col_cluster=True,
    row_cluster=True,
    linewidths=0.1,
    cbar_kws={'label': 'Standardized Value'},
    dendrogram_ratio=0.15
)
plt.suptitle('Mean Features Clustermap\n(Blue=Benign, Red=Malignant)', y=0.95, fontsize=14)
plt.savefig('mean_features_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Worst features clustermap
plt.figure(figsize=(12, 10))
X_worst_scaled = StandardScaler().fit_transform(X_worst)
X_worst_scaled_df = pd.DataFrame(X_worst_scaled, columns=X_worst.columns, index=X_worst.index)

worst_clustermap = sns.clustermap(
    X_worst_scaled_df,
    figsize=(12, 10),
    cmap='RdYlBu_r',
    row_colors=diagnosis_colors,
    col_cluster=True,
    row_cluster=True,
    linewidths=0.1,
    cbar_kws={'label': 'Standardized Value'},
    dendrogram_ratio=0.15
)
plt.suptitle('Worst Features Clustermap\n(Blue=Benign, Red=Malignant)', y=0.95, fontsize=14)
plt.savefig('worst_features_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Correlation clustermap with feature labels
plt.figure(figsize=(16, 14))
correlation_matrix = X_scaled_df.corr()

print("\n" + "="*60)
print("FEATURE CORRELATION CLUSTERMAP - ALL FEATURES")
print("="*60)

print(f"\nTotal number of features: {len(correlation_matrix.columns)}")
print("\nAll features in the correlation matrix:")
for i, feature in enumerate(correlation_matrix.columns, 1):
    print(f"{i:2d}. {feature}")

print(f"\nFeature categories breakdown:")
mean_count = sum(1 for col in correlation_matrix.columns if col.endswith('_me'))
se_count = sum(1 for col in correlation_matrix.columns if col.endswith('_se'))
worst_count = sum(1 for col in correlation_matrix.columns if col.endswith('_wo'))
print(f"  Mean features (_me): {mean_count}")
print(f"  SE features (_se): {se_count}")
print(f"  Worst features (_wo): {worst_count}")

# Create feature type colors for better visualization
feature_colors = []
for feature in correlation_matrix.columns:
    if feature.endswith('_me'):
        feature_colors.append('lightgreen')
    elif feature.endswith('_se'):
        feature_colors.append('lightyellow')
    elif feature.endswith('_wo'):
        feature_colors.append('lightpink')
    else:
        feature_colors.append('lightgray')

corr_clustermap = sns.clustermap(
    correlation_matrix,
    figsize=(16, 14),
    cmap='coolwarm',
    center=0,
    square=False,
    linewidths=0.1,
    cbar_kws={'label': 'Correlation Coefficient'},
    dendrogram_ratio=0.1,
    annot=False,  # Set to True if you want correlation values displayed
    row_colors=feature_colors,
    col_colors=feature_colors,
    xticklabels=True,  # Show feature names on x-axis
    yticklabels=True,  # Show feature names on y-axis
    colors_ratio=0.02
)

# Improve readability of feature labels
corr_clustermap.ax_heatmap.set_xticklabels(
    corr_clustermap.ax_heatmap.get_xticklabels(), 
    rotation=45, 
    ha='right', 
    fontsize=8
)
corr_clustermap.ax_heatmap.set_yticklabels(
    corr_clustermap.ax_heatmap.get_yticklabels(), 
    rotation=0, 
    fontsize=8
)

plt.suptitle('Feature Correlation Clustermap\n(Green=Mean, Yellow=SE, Pink=Worst)', y=0.95, fontsize=12)
plt.savefig('correlation_clustermap_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation matrix statistics
print(f"\nCorrelation Matrix Statistics:")
print(f"  Matrix shape: {correlation_matrix.shape}")
print(f"  Highest correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max():.4f}")
print(f"  Lowest correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min():.4f}")
print(f"  Mean correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.4f}")

# Find and print the highest correlated feature pairs
print(f"\nTop 10 highest correlated feature pairs:")
# Get upper triangle of correlation matrix
mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
correlation_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        feature1 = correlation_matrix.columns[i]
        feature2 = correlation_matrix.columns[j]
        correlation_pairs.append((abs(corr_value), corr_value, feature1, feature2))

# Sort by absolute correlation value
correlation_pairs.sort(reverse=True)

for i, (abs_corr, corr, feat1, feat2) in enumerate(correlation_pairs[:10], 1):
    print(f"{i:2d}. {feat1} ↔ {feat2}: {corr:+.4f}")

print(f"\nFeature order in the clustermap (after clustering):")
clustered_columns = corr_clustermap.dendrogram_col.reordered_ind
for i, col_idx in enumerate(clustered_columns, 1):
    feature_name = correlation_matrix.columns[col_idx]
    print(f"{i:2d}. {feature_name}")

print(f"\nColor legend for feature types:")
print(f"  🟢 Light Green: Mean features (_me)")
print(f"  🟡 Light Yellow: Standard Error features (_se)")
print(f"  🩷 Light Pink: Worst features (_wo)")

print("Generated clustermap files:")
print("- full_dataset_clustermap.png")
print("- mean_features_clustermap.png") 
print("- worst_features_clustermap.png")
print("- correlation_clustermap.png")




