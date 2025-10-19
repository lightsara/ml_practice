
##imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import skewnorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Prep data
data_file = open("/home/saran/main/code/MLKTH/exercises/Project/data/data_kaggle_noletters.csv", "r")
df = pd.read_csv(data_file, sep=",", index_col=["id"])
data_file.close()

##Check for N/As
null_matrix = df.isnull()
nollfinns = 0
for i in null_matrix:
    if True in null_matrix[i]:
        nollfinns = 1

# Print statement
print("Are there any 0s? ", nollfinns)

# Replace non-numeric numbers. (did manually)

# Plot the heatmap
plt.rcParams.update({'font.size': 30})
f = plt.figure()
f.set_figwidth(31)
f.set_figheight(31)
correl = df[1:].corr()

#print(correl)

# Generate a mask for the upper triangle maska =
#np.triu(np.ones_like(correl, dtype=bool)) ##mask out upper right
#quadrant.

heatmap_plot = sns.heatmap(correl, cmap="Greys") ##, mask=maska)
heatmap_fig = heatmap_plot.get_figure()
heatmap_fig.savefig("heatmap_fig.png")

# Prepare data for machine learning
# Separate features and target
X = df.drop('diagnosis', axis=1)  # Features (all columns except diagnosis)
y = df['diagnosis']  # Target variable (diagnosis)

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# PCA Analysis
print("\n=== Principal Component Analysis ===")

# Perform PCA on the full dataset to understand explained variance
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Calculate cumulative explained variance
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Total features: {len(explained_variance_ratio)}")
print(f"Explained variance by first 5 components: {explained_variance_ratio[:5]}")
print(f"Cumulative variance by first 5 components: {cumulative_variance[:5]}")

# Find number of components for different variance thresholds
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"Components needed for {threshold*100:.0f}% variance: {n_components}")

# Create PCA visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Explained variance by component
axes[0, 0].bar(range(1, min(21, len(explained_variance_ratio) + 1)), 
               explained_variance_ratio[:20], alpha=0.7)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Explained Variance by Principal Component (First 20)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cumulative explained variance
axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: PCA 2D visualization
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
scatter = axes[1, 0].scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], 
                             c=y_train, cmap='viridis', alpha=0.6)
axes[1, 0].set_xlabel(f'First PC ({pca_2d.explained_variance_ratio_[0]:.3f} variance)')
axes[1, 0].set_ylabel(f'Second PC ({pca_2d.explained_variance_ratio_[1]:.3f} variance)')
axes[1, 0].set_title('2D PCA Projection')
plt.colorbar(scatter, ax=axes[1, 0])

# Plot 4: PCA 3D setup (we'll show the first 3 components in different way)
pca_3d = PCA(n_components=3)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)
scatter2 = axes[1, 1].scatter(X_train_pca_3d[:, 0], X_train_pca_3d[:, 2], 
                              c=y_train, cmap='viridis', alpha=0.6)
axes[1, 1].set_xlabel(f'First PC ({pca_3d.explained_variance_ratio_[0]:.3f} variance)')
axes[1, 1].set_ylabel(f'Third PC ({pca_3d.explained_variance_ratio_[2]:.3f} variance)')
axes[1, 1].set_title('PCA: 1st vs 3rd Component')
plt.colorbar(scatter2, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Test different numbers of PCA components for classification performance
print("\n=== PCA + Classification Performance ===")
pca_components = [2, 5, 10, 15, 20, 25, 30]  # 30 is original dimension
pca_results = {}

for n_comp in pca_components:
    if n_comp > X_train_scaled.shape[1]:
        continue
        
    print(f"\nTesting with {n_comp} PCA components:")
    
    # Apply PCA
    if n_comp == 30:  # Original dimensions
        X_train_pca = X_train_scaled
        X_test_pca = X_test_scaled
        variance_explained = 1.0
    else:
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        variance_explained = np.sum(pca.explained_variance_ratio_)
    
    print(f"  Variance explained: {variance_explained:.4f}")
    
    # Test Logistic Regression with PCA
    lr_pca = LogisticRegression(random_state=42, max_iter=1000)
    lr_pca.fit(X_train_pca, y_train)
    y_pred_lr_pca = lr_pca.predict(X_test_pca)
    accuracy_lr_pca = accuracy_score(y_test, y_pred_lr_pca)
    
    # Test KNN with PCA
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_knn_pca = knn_pca.predict(X_test_pca)
    accuracy_knn_pca = accuracy_score(y_test, y_pred_knn_pca)
    
    # Test K-Means with PCA
    kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
    train_clusters_pca = kmeans_pca.fit_predict(X_train_pca)
    
    # Create cluster-to-label mapping
    cluster_labels_pca = {}
    for cluster_id in range(5):
        cluster_mask = train_clusters_pca == cluster_id
        cluster_true_labels = y_train[cluster_mask]
        if len(cluster_true_labels) > 0:
            most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
            cluster_labels_pca[cluster_id] = most_common_label
    
    test_clusters_pca = kmeans_pca.predict(X_test_pca)
    y_pred_kmeans_pca = np.array([cluster_labels_pca.get(cluster_id, 0) for cluster_id in test_clusters_pca])
    accuracy_kmeans_pca = accuracy_score(y_test, y_pred_kmeans_pca)
    
    print(f"  Logistic Regression accuracy: {accuracy_lr_pca:.4f}")
    print(f"  KNN accuracy: {accuracy_knn_pca:.4f}")
    print(f"  K-Means accuracy: {accuracy_kmeans_pca:.4f}")
    
    pca_results[n_comp] = {
        'lr_accuracy': accuracy_lr_pca,
        'knn_accuracy': accuracy_knn_pca,
        'kmeans_accuracy': accuracy_kmeans_pca,
        'variance_explained': variance_explained
    }

# Initialize and train Logistic Regression.
print("\n=== Logistic Regression ===")
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Make predictions with Logistic Regression.
y_pred_lr = logreg.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Initialize and train KNN with different k values.
print("\n=== K-Nearest Neighbors ===")
k_values = [3, 5, 7, 9, 11]
knn_results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_knn = knn.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    
    knn_results[k] = {
        'model': knn,
        'accuracy': accuracy_knn,
        'predictions': y_pred_knn
    }
    
    print(f"KNN (k={k}) Accuracy: {accuracy_knn:.4f}")

# Find the best k value.
best_k = max(knn_results.keys(), key=lambda k: knn_results[k]['accuracy'])
best_knn_accuracy = knn_results[best_k]['accuracy']
best_knn_predictions = knn_results[best_k]['predictions']

print(f"\nBest KNN model: k={best_k} with accuracy: {best_knn_accuracy:.4f}")
print("Best KNN Classification Report:")
print(classification_report(y_test, best_knn_predictions))

# DBSCAN Clustering for Classification
print("\n=== DBSCAN Clustering ===")

# Try different eps values to find optimal clustering
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
dbscan_results = {}

for eps in eps_values:
    # Apply DBSCAN clustering on training data
    dbscan = DBSCAN(eps=eps, min_samples=5)
    train_clusters = dbscan.fit_predict(X_train_scaled)
    
    # Count unique clusters (excluding noise points labeled as -1)
    unique_clusters = set(train_clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise = list(train_clusters).count(-1)
    
    print(f"DBSCAN (eps={eps}): {n_clusters} clusters, {n_noise} noise points")
    
    if n_clusters > 1 and n_clusters < len(y_train):  # Valid clustering
        # Calculate silhouette score (excluding noise points)
        if n_noise < len(train_clusters):
            valid_indices = train_clusters != -1
            if np.sum(valid_indices) > 1:
                silhouette_avg = silhouette_score(X_train_scaled[valid_indices], 
                                                train_clusters[valid_indices])
                print(f"  Silhouette Score: {silhouette_avg:.4f}")
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
            
        # Create cluster-to-label mapping based on majority vote in each cluster
        cluster_labels = {}
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise points
                cluster_mask = train_clusters == cluster_id
                cluster_true_labels = y_train[cluster_mask]
                if len(cluster_true_labels) > 0:
                    # Assign the most common label in this cluster
                    most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
                    cluster_labels[cluster_id] = most_common_label
        
        # Predict on test set
        test_clusters = dbscan.fit_predict(X_test_scaled)
        y_pred_dbscan = np.full(len(test_clusters), -1)  # Initialize with -1 (unknown)
        
        for i, cluster_id in enumerate(test_clusters):
            if cluster_id in cluster_labels:
                y_pred_dbscan[i] = cluster_labels[cluster_id]
            else:
                # For noise points or unknown clusters, assign the most common class
                y_pred_dbscan[i] = Counter(y_train).most_common(1)[0][0]
        
        # Calculate accuracy only for points that got valid predictions
        valid_predictions = y_pred_dbscan != -1
        if np.sum(valid_predictions) > 0:
            accuracy_dbscan = accuracy_score(y_test[valid_predictions], 
                                           y_pred_dbscan[valid_predictions])
            print(f"  Classification Accuracy: {accuracy_dbscan:.4f}")
            
            dbscan_results[eps] = {
                'model': dbscan,
                'accuracy': accuracy_dbscan,
                'predictions': y_pred_dbscan,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette_avg,
                'cluster_labels': cluster_labels
            }
        else:
            print(f"  No valid predictions possible")
    else:
        print(f"  Invalid clustering (too few or too many clusters)")

# Find best DBSCAN configuration
if dbscan_results:
    best_eps = max(dbscan_results.keys(), key=lambda eps: dbscan_results[eps]['accuracy'])
    best_dbscan_accuracy = dbscan_results[best_eps]['accuracy']
    best_dbscan_predictions = dbscan_results[best_eps]['predictions']
    best_dbscan_clusters = dbscan_results[best_eps]['n_clusters']
    
    print(f"\nBest DBSCAN model: eps={best_eps} with accuracy: {best_dbscan_accuracy:.4f}")
    print(f"Best DBSCAN found {best_dbscan_clusters} clusters")
    print("Best DBSCAN Classification Report:")
    valid_mask = best_dbscan_predictions != -1
    if np.sum(valid_mask) > 0:
        print(classification_report(y_test[valid_mask], best_dbscan_predictions[valid_mask]))
else:
    print("No valid DBSCAN configurations found")
    best_dbscan_accuracy = 0
    best_eps = None

# K-Means Clustering for Classification
print("\n=== K-Means Clustering ===")

# Try different numbers of clusters
k_clusters = [2, 3, 4, 5, 6]
kmeans_results = {}

for k in k_clusters:
    # Apply K-Means clustering on training data
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_train_scaled, train_clusters)
    print(f"K-Means (k={k}): Silhouette Score: {silhouette_avg:.4f}")
    
    # Create cluster-to-label mapping based on majority vote in each cluster
    cluster_labels = {}
    for cluster_id in range(k):
        cluster_mask = train_clusters == cluster_id
        cluster_true_labels = y_train[cluster_mask]
        if len(cluster_true_labels) > 0:
            # Assign the most common label in this cluster
            most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
            cluster_labels[cluster_id] = most_common_label
    
    # Predict on test set
    test_clusters = kmeans.predict(X_test_scaled)
    y_pred_kmeans = np.array([cluster_labels.get(cluster_id, 0) for cluster_id in test_clusters])
    
    # Calculate accuracy
    accuracy_kmeans = accuracy_score(y_test, y_pred_kmeans)
    print(f"  Classification Accuracy: {accuracy_kmeans:.4f}")
    
    kmeans_results[k] = {
        'model': kmeans,
        'accuracy': accuracy_kmeans,
        'predictions': y_pred_kmeans,
        'silhouette': silhouette_avg,
        'cluster_labels': cluster_labels,
        'inertia': kmeans.inertia_
    }

# Find best K-Means configuration
best_k_kmeans = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['accuracy'])
best_kmeans_accuracy = kmeans_results[best_k_kmeans]['accuracy']
best_kmeans_predictions = kmeans_results[best_k_kmeans]['predictions']
best_kmeans_silhouette = kmeans_results[best_k_kmeans]['silhouette']

print(f"\nBest K-Means model: k={best_k_kmeans} with accuracy: {best_kmeans_accuracy:.4f}")
print(f"Best K-Means Silhouette Score: {best_kmeans_silhouette:.4f}")
print("Best K-Means Classification Report:")
print(classification_report(y_test, best_kmeans_predictions))

# Hierarchical Clustering for Classification
print("\n=== Hierarchical Clustering ===")

# Try different numbers of clusters and linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters_hier = [2, 3, 4, 5, 6]
hierarchical_results = {}

for linkage_method in linkage_methods:
    print(f"\nTesting {linkage_method} linkage:")
    linkage_results = {}
    
    for n_clusters in n_clusters_hier:
        try:
            # Apply Agglomerative Clustering on training data
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            train_clusters = hierarchical.fit_predict(X_train_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_train_scaled, train_clusters)
            
            # Create cluster-to-label mapping based on majority vote in each cluster
            cluster_labels = {}
            for cluster_id in range(n_clusters):
                cluster_mask = train_clusters == cluster_id
                cluster_true_labels = y_train[cluster_mask]
                if len(cluster_true_labels) > 0:
                    # Assign the most common label in this cluster
                    most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
                    cluster_labels[cluster_id] = most_common_label
            
            # For test set, we need to fit and predict again since AgglomerativeClustering
            # doesn't have a predict method
            hierarchical_test = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            test_clusters = hierarchical_test.fit_predict(X_test_scaled)
            y_pred_hierarchical = np.array([cluster_labels.get(cluster_id, 0) for cluster_id in test_clusters])
            
            # Calculate accuracy
            accuracy_hierarchical = accuracy_score(y_test, y_pred_hierarchical)
            
            print(f"  n_clusters={n_clusters}: Silhouette={silhouette_avg:.4f}, Accuracy={accuracy_hierarchical:.4f}")
            
            linkage_results[n_clusters] = {
                'model': hierarchical,
                'accuracy': accuracy_hierarchical,
                'predictions': y_pred_hierarchical,
                'silhouette': silhouette_avg,
                'cluster_labels': cluster_labels
            }
            
        except Exception as e:
            print(f"  n_clusters={n_clusters}: Error - {str(e)}")
    
    if linkage_results:
        hierarchical_results[linkage_method] = linkage_results

# Find best Hierarchical Clustering configuration
if hierarchical_results:
    best_config = None
    best_hierarchical_accuracy = 0
    
    for linkage_method in hierarchical_results:
        for n_clusters in hierarchical_results[linkage_method]:
            accuracy = hierarchical_results[linkage_method][n_clusters]['accuracy']
            if accuracy > best_hierarchical_accuracy:
                best_hierarchical_accuracy = accuracy
                best_config = (linkage_method, n_clusters)
    
    if best_config:
        best_linkage, best_n_clusters_hier = best_config
        best_hierarchical_predictions = hierarchical_results[best_linkage][best_n_clusters_hier]['predictions']
        best_hierarchical_silhouette = hierarchical_results[best_linkage][best_n_clusters_hier]['silhouette']
        
        print(f"\nBest Hierarchical model: {best_linkage} linkage, n_clusters={best_n_clusters_hier}")
        print(f"Best Hierarchical Accuracy: {best_hierarchical_accuracy:.4f}")
        print(f"Best Hierarchical Silhouette Score: {best_hierarchical_silhouette:.4f}")
        print("Best Hierarchical Classification Report:")
        print(classification_report(y_test, best_hierarchical_predictions))
    else:
        best_hierarchical_accuracy = 0
        best_config = None
else:
    print("No valid Hierarchical Clustering configurations found")
    best_hierarchical_accuracy = 0
    best_config = None

# Compare models
print("\n=== Model Comparison ===")
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print(f"Best KNN (k={best_k}) Accuracy: {best_knn_accuracy:.4f}")
if dbscan_results:
    print(f"Best DBSCAN (eps={best_eps}) Accuracy: {best_dbscan_accuracy:.4f}")
print(f"Best K-Means (k={best_k_kmeans}) Accuracy: {best_kmeans_accuracy:.4f}")
if hierarchical_results and best_config:
    print(f"Best Hierarchical ({best_linkage}, n={best_n_clusters_hier}) Accuracy: {best_hierarchical_accuracy:.4f}")

# Determine best performing model
accuracies = {
    'Logistic Regression': accuracy_lr,
    'KNN': best_knn_accuracy,
    'K-Means': best_kmeans_accuracy,
}
if dbscan_results:
    accuracies['DBSCAN'] = best_dbscan_accuracy
if hierarchical_results and best_config:
    accuracies['Hierarchical'] = best_hierarchical_accuracy

best_model = max(accuracies.keys(), key=lambda k: accuracies[k])
print(f"\nBest performing model: {best_model} with accuracy: {accuracies[best_model]:.4f}")

# Plot KNN performance for different k values
plt.figure(figsize=(10, 6))
k_vals = list(knn_results.keys())
knn_accuracies = [knn_results[k]['accuracy'] for k in k_vals]

plt.plot(k_vals, knn_accuracies, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs Number of Neighbors')
plt.grid(True, alpha=0.3)
plt.xticks(k_vals)
plt.savefig('knn_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot DBSCAN performance for different eps values
if dbscan_results:
    plt.figure(figsize=(10, 6))
    eps_vals = list(dbscan_results.keys())
    dbscan_accuracies = [dbscan_results[eps]['accuracy'] for eps in eps_vals]
    
    plt.plot(eps_vals, dbscan_accuracies, marker='s', linewidth=2, markersize=8, color='red')
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Accuracy')
    plt.title('DBSCAN Performance vs Epsilon Parameter')
    plt.grid(True, alpha=0.3)
    plt.xticks(eps_vals)
    plt.savefig('dbscan_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot K-Means performance for different k values
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy vs Number of Clusters
plt.subplot(1, 2, 1)
k_cluster_vals = list(kmeans_results.keys())
kmeans_accuracies = [kmeans_results[k]['accuracy'] for k in k_cluster_vals]

plt.plot(k_cluster_vals, kmeans_accuracies, marker='d', linewidth=2, markersize=8, color='purple')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Accuracy')
plt.title('K-Means Accuracy vs Number of Clusters')
plt.grid(True, alpha=0.3)
plt.xticks(k_cluster_vals)

# Subplot 2: Elbow Method - Inertia vs Number of Clusters
plt.subplot(1, 2, 2)
inertias = [kmeans_results[k]['inertia'] for k in k_cluster_vals]

plt.plot(k_cluster_vals, inertias, marker='d', linewidth=2, markersize=8, color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('K-Means Elbow Method')
plt.grid(True, alpha=0.3)
plt.xticks(k_cluster_vals)

plt.tight_layout()
plt.savefig('kmeans_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot Hierarchical Clustering performance
if hierarchical_results:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'green', 'red', 'orange']
    linkage_color_map = dict(zip(linkage_methods, colors))
    
    for i, linkage_method in enumerate(hierarchical_results):
        row, col = i // 2, i % 2
        
        n_clusters_list = list(hierarchical_results[linkage_method].keys())
        accuracies_hier = [hierarchical_results[linkage_method][k]['accuracy'] for k in n_clusters_list]
        
        axes[row, col].plot(n_clusters_list, accuracies_hier, marker='o', linewidth=2, 
                           markersize=8, color=linkage_color_map[linkage_method])
        axes[row, col].set_xlabel('Number of Clusters')
        axes[row, col].set_ylabel('Accuracy')
        axes[row, col].set_title(f'Hierarchical Clustering: {linkage_method.capitalize()} Linkage')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xticks(n_clusters_list)
    
    plt.tight_layout()
    plt.savefig('hierarchical_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# PCA Performance Analysis Plots
print("\n=== PCA Performance Visualization ===")

# Plot PCA + Classification Performance
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy vs PCA Components
components = list(pca_results.keys())
lr_accuracies = [pca_results[comp]['lr_accuracy'] for comp in components]
knn_accuracies = [pca_results[comp]['knn_accuracy'] for comp in components]
kmeans_accuracies = [pca_results[comp]['kmeans_accuracy'] for comp in components]

axes[0, 0].plot(components, lr_accuracies, 'o-', label='Logistic Regression', linewidth=2, markersize=6)
axes[0, 0].plot(components, knn_accuracies, 's-', label='KNN', linewidth=2, markersize=6)
axes[0, 0].plot(components, kmeans_accuracies, '^-', label='K-Means', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Number of PCA Components')
axes[0, 0].set_ylabel('Classification Accuracy')
axes[0, 0].set_title('Classification Performance vs PCA Components')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Variance Explained vs Components
variances = [pca_results[comp]['variance_explained'] for comp in components]
axes[0, 1].plot(components, variances, 'ro-', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Number of PCA Components')
axes[0, 1].set_ylabel('Cumulative Variance Explained')
axes[0, 1].set_title('Variance Explained vs PCA Components')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Accuracy vs Variance Trade-off
axes[1, 0].scatter(variances, lr_accuracies, c=components, cmap='viridis', s=60, alpha=0.7, label='Logistic Regression')
axes[1, 0].scatter(variances, knn_accuracies, c=components, cmap='plasma', s=60, alpha=0.7, marker='s', label='KNN')
axes[1, 0].set_xlabel('Variance Explained')
axes[1, 0].set_ylabel('Classification Accuracy')
axes[1, 0].set_title('Accuracy vs Variance Trade-off')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Component importance (first 15 components)
pca_importance = PCA(n_components=15)
pca_importance.fit(X_train_scaled)
importance_ratios = pca_importance.explained_variance_ratio_
axes[1, 1].bar(range(1, len(importance_ratios) + 1), importance_ratios, alpha=0.7, color='skyblue')
axes[1, 1].set_xlabel('Principal Component')
axes[1, 1].set_ylabel('Explained Variance Ratio')
axes[1, 1].set_title('Individual Component Importance (First 15)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Find optimal number of components
print("\n=== PCA Optimization Results ===")
best_lr_comp = max(components, key=lambda comp: pca_results[comp]['lr_accuracy'])
best_knn_comp = max(components, key=lambda comp: pca_results[comp]['knn_accuracy'])
best_kmeans_comp = max(components, key=lambda comp: pca_results[comp]['kmeans_accuracy'])

print(f"Best PCA components for Logistic Regression: {best_lr_comp} ({pca_results[best_lr_comp]['lr_accuracy']:.4f} accuracy)")
print(f"Best PCA components for KNN: {best_knn_comp} ({pca_results[best_knn_comp]['knn_accuracy']:.4f} accuracy)")
print(f"Best PCA components for K-Means: {best_kmeans_comp} ({pca_results[best_kmeans_comp]['kmeans_accuracy']:.4f} accuracy)")

# Compare with original performance
print(f"\nOriginal vs PCA Performance:")
print(f"Logistic Regression: {accuracy_lr:.4f} (original) vs {pca_results[best_lr_comp]['lr_accuracy']:.4f} (PCA-{best_lr_comp})")
print(f"KNN: {best_knn_accuracy:.4f} (original) vs {pca_results[best_knn_comp]['knn_accuracy']:.4f} (PCA-{best_knn_comp})")
print(f"K-Means: {best_kmeans_accuracy:.4f} (original) vs {pca_results[best_kmeans_comp]['kmeans_accuracy']:.4f} (PCA-{best_kmeans_comp})")

# Feature Importance Analysis using PCA
print("\n=== Feature Importance via PCA ===")

# Get feature names
feature_names = X.columns.tolist()

# Analyze the first few principal components
pca_feature_analysis = PCA(n_components=5)
pca_feature_analysis.fit(X_train_scaled)

# Create feature importance heatmap
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: PCA Components Heatmap
components_df = pd.DataFrame(
    pca_feature_analysis.components_[:5],
    columns=feature_names,
    index=[f'PC{i+1} ({pca_feature_analysis.explained_variance_ratio_[i]:.3f})' for i in range(5)]
)

sns.heatmap(components_df, annot=False, cmap='RdBu_r', center=0, ax=axes[0])
axes[0].set_title('PCA Components: Feature Loadings\n(Red: Positive, Blue: Negative contribution)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Principal Components')

# Plot 2: Top contributing features for first 2 PCs
pc1_contributions = np.abs(pca_feature_analysis.components_[0])
pc2_contributions = np.abs(pca_feature_analysis.components_[1])

# Get top 10 features for each PC
pc1_top_idx = np.argsort(pc1_contributions)[-10:]
pc2_top_idx = np.argsort(pc2_contributions)[-10:]

y_pos1 = np.arange(len(pc1_top_idx))
y_pos2 = np.arange(len(pc2_top_idx))

# Create subplot for top features
axes[1].barh(y_pos1, pc1_contributions[pc1_top_idx], alpha=0.7, label='PC1', color='blue')
axes[1].barh(y_pos2 + 0.4, pc2_contributions[pc2_top_idx], alpha=0.7, label='PC2', color='red')

# Set labels
pc1_labels = [feature_names[i] for i in pc1_top_idx]
pc2_labels = [feature_names[i] for i in pc2_top_idx]

axes[1].set_yticks(y_pos1)
axes[1].set_yticklabels(pc1_labels, fontsize=8)
axes[1].set_xlabel('Absolute Loading Value')
axes[1].set_title('Top 10 Contributing Features for PC1 and PC2')
axes[1].legend()

plt.tight_layout()
plt.savefig('pca_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top contributing features
print("Top 5 features contributing to PC1:")
pc1_top5_idx = np.argsort(np.abs(pca_feature_analysis.components_[0]))[-5:]
for i, idx in enumerate(reversed(pc1_top5_idx)):
    print(f"  {i+1}. {feature_names[idx]}: {pca_feature_analysis.components_[0][idx]:.4f}")

print("\nTop 5 features contributing to PC2:")
pc2_top5_idx = np.argsort(np.abs(pca_feature_analysis.components_[1]))[-5:]
for i, idx in enumerate(reversed(pc2_top5_idx)):
    print(f"  {i+1}. {feature_names[idx]}: {pca_feature_analysis.components_[1][idx]:.4f}")

# Create a dendrogram (sample of data for readability)
print("\nGenerating dendrogram...")
# Use a sample of training data for dendrogram (full dataset would be too cluttered)
sample_size = min(100, len(X_train_scaled))
sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_sample = X_train_scaled[sample_indices]
y_sample = y_train.iloc[sample_indices]

# Compute linkage matrix
linkage_matrix = linkage(X_sample, method='ward')

# Create dendrogram
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, 
           truncate_mode='level', 
           p=5,  # Show only top 5 levels
           leaf_rotation=90,
           leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)\nSample of 100 data points')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance')
plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Silhouette scores comparison for clustering methods
plt.figure(figsize=(12, 8))

# K-Means silhouette scores
kmeans_silhouettes = [kmeans_results[k]['silhouette'] for k in k_cluster_vals]
plt.plot(k_cluster_vals, kmeans_silhouettes, marker='d', linewidth=2, markersize=8, 
         color='purple', label='K-Means')

# Hierarchical clustering silhouette scores (best linkage method)
if hierarchical_results and best_config:
    best_linkage, _ = best_config
    hier_n_clusters = list(hierarchical_results[best_linkage].keys())
    hier_silhouettes = [hierarchical_results[best_linkage][k]['silhouette'] for k in hier_n_clusters]
    plt.plot(hier_n_clusters, hier_silhouettes, marker='^', linewidth=2, markersize=8, 
             color='brown', label=f'Hierarchical ({best_linkage})')

# DBSCAN silhouette scores (if available)
if dbscan_results:
    dbscan_eps_vals = []
    dbscan_silhouettes = []
    for eps in dbscan_results:
        if dbscan_results[eps]['silhouette'] > -1:  # Valid silhouette score
            dbscan_eps_vals.append(eps)
            dbscan_silhouettes.append(dbscan_results[eps]['silhouette'])
    
    if dbscan_silhouettes:
        # Create a secondary x-axis for DBSCAN
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.plot(dbscan_eps_vals, dbscan_silhouettes, marker='s', linewidth=2, markersize=8, 
                 color='red', label='DBSCAN')
        ax2.set_xlabel('DBSCAN Epsilon', color='red')
        ax2.tick_params(axis='x', labelcolor='red')

plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Clustering Quality Comparison: Silhouette Scores')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.savefig('clustering_silhouette_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Model comparison bar chart
plt.figure(figsize=(10, 6))
model_names = list(accuracies.keys())
model_accuracies = list(accuracies.values())
colors = ['blue', 'green', 'red'][:len(model_names)]

bars = plt.bar(model_names, model_accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)

# Add accuracy values on top of bars
for bar, acc in zip(bars, model_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrices
# Determine number of models to display
n_models = 3  # Base: LR, KNN, K-Means
if dbscan_results:
    n_models += 1
if hierarchical_results and best_config:
    n_models += 1

# Create appropriate subplot grid
if n_models <= 4:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

plot_idx = 0

# Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx])
axes[plot_idx].set_title(f'Logistic Regression\nAccuracy: {accuracy_lr:.4f}')
axes[plot_idx].set_xlabel('Predicted')
axes[plot_idx].set_ylabel('Actual')
plot_idx += 1

# Best KNN confusion matrix
cm_knn = confusion_matrix(y_test, best_knn_predictions)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[plot_idx])
axes[plot_idx].set_title(f'KNN (k={best_k})\nAccuracy: {best_knn_accuracy:.4f}')
axes[plot_idx].set_xlabel('Predicted')
axes[plot_idx].set_ylabel('Actual')
plot_idx += 1

# K-Means confusion matrix
cm_kmeans = confusion_matrix(y_test, best_kmeans_predictions)
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Purples', ax=axes[plot_idx])
axes[plot_idx].set_title(f'K-Means (k={best_k_kmeans})\nAccuracy: {best_kmeans_accuracy:.4f}')
axes[plot_idx].set_xlabel('Predicted')
axes[plot_idx].set_ylabel('Actual')
plot_idx += 1

# Hierarchical confusion matrix (if available)
if hierarchical_results and best_config:
    cm_hierarchical = confusion_matrix(y_test, best_hierarchical_predictions)
    sns.heatmap(cm_hierarchical, annot=True, fmt='d', cmap='Oranges', ax=axes[plot_idx])
    axes[plot_idx].set_title(f'Hierarchical ({best_linkage}, n={best_n_clusters_hier})\nAccuracy: {best_hierarchical_accuracy:.4f}')
    axes[plot_idx].set_xlabel('Predicted')
    axes[plot_idx].set_ylabel('Actual')
    plot_idx += 1

# DBSCAN confusion matrix (if available)
if dbscan_results:
    valid_mask = best_dbscan_predictions != -1
    if np.sum(valid_mask) > 0:
        cm_dbscan = confusion_matrix(y_test[valid_mask], best_dbscan_predictions[valid_mask])
        sns.heatmap(cm_dbscan, annot=True, fmt='d', cmap='Reds', ax=axes[plot_idx])
        axes[plot_idx].set_title(f'DBSCAN (eps={best_eps})\nAccuracy: {best_dbscan_accuracy:.4f}')
        axes[plot_idx].set_xlabel('Predicted')
        axes[plot_idx].set_ylabel('Actual')
        plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()


