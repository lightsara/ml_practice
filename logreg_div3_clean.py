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

# Divide features into three categories based on suffixes
print("\n=== Dataset Division ===")

# Get all column names
all_features = X.columns.tolist()

# Divide features by categories
mean_features = [col for col in all_features if col.endswith('_mean')]
se_features = [col for col in all_features if col.endswith('_se')]
worst_features = [col for col in all_features if col.endswith('_worst')]

print(f"Mean features ({len(mean_features)}): {mean_features}")
print(f"Standard Error features ({len(se_features)}): {se_features}")
print(f"Worst features ({len(worst_features)}): {worst_features}")

# Create separate datasets for each category
X_mean = X[mean_features]
X_se = X[se_features]
X_worst = X[worst_features]

print(f"\nMean dataset shape: {X_mean.shape}")
print(f"SE dataset shape: {X_se.shape}")
print(f"Worst dataset shape: {X_worst.shape}")

# Split each dataset into training and testing sets
X_mean_train, X_mean_test, y_train, y_test = train_test_split(X_mean, y, test_size=0.2, random_state=42, stratify=y)
X_se_train, X_se_test, _, _ = train_test_split(X_se, y, test_size=0.2, random_state=42, stratify=y)
X_worst_train, X_worst_test, _, _ = train_test_split(X_worst, y, test_size=0.2, random_state=42, stratify=y)

# Also keep the full dataset
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for each dataset
print("\n=== Feature Scaling ===")

# Scale full dataset
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train)
X_test_scaled = scaler_full.transform(X_test)

# Scale mean features
scaler_mean = StandardScaler()
X_mean_train_scaled = scaler_mean.fit_transform(X_mean_train)
X_mean_test_scaled = scaler_mean.transform(X_mean_test)

# Scale SE features
scaler_se = StandardScaler()
X_se_train_scaled = scaler_se.fit_transform(X_se_train)
X_se_test_scaled = scaler_se.transform(X_se_test)

# Scale worst features
scaler_worst = StandardScaler()
X_worst_train_scaled = scaler_worst.fit_transform(X_worst_train)
X_worst_test_scaled = scaler_worst.transform(X_worst_test)

print(f"Full dataset - Training: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Mean dataset - Training: {X_mean_train_scaled.shape}, Test: {X_mean_test_scaled.shape}")
print(f"SE dataset - Training: {X_se_train_scaled.shape}, Test: {X_se_test_scaled.shape}")
print(f"Worst dataset - Training: {X_worst_train_scaled.shape}, Test: {X_worst_test_scaled.shape}")

# Function to run classification analysis on a dataset
def run_classification_analysis(X_train_data, X_test_data, y_train_data, y_test_data, dataset_name):
    """Run comprehensive classification analysis on a given dataset"""
    print(f"\n=== {dataset_name} Dataset Analysis ===")
    
    results = {}
    
    # Logistic Regression
    print(f"\n{dataset_name} - Logistic Regression:")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train_data, y_train_data)
    y_pred_lr = logreg.predict(X_test_data)
    accuracy_lr = accuracy_score(y_test_data, y_pred_lr)
    print(f"  Accuracy: {accuracy_lr:.4f}")
    
    results['lr'] = {
        'model': logreg,
        'predictions': y_pred_lr,
        'accuracy': accuracy_lr
    }
    
    # KNN
    print(f"\n{dataset_name} - K-Nearest Neighbors:")
    best_knn_acc = 0
    best_knn_k = 0
    best_knn_pred = None
    
    for k in [3, 5, 7, 9, 11]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_data, y_train_data)
        y_pred_knn = knn.predict(X_test_data)
        accuracy_knn = accuracy_score(y_test_data, y_pred_knn)
        print(f"  k={k}: {accuracy_knn:.4f}")
        
        if accuracy_knn > best_knn_acc:
            best_knn_acc = accuracy_knn
            best_knn_k = k
            best_knn_pred = y_pred_knn
    
    results['knn'] = {
        'best_k': best_knn_k,
        'predictions': best_knn_pred,
        'accuracy': best_knn_acc
    }
    
    # K-Means Clustering
    print(f"\n{dataset_name} - K-Means Clustering:")
    best_kmeans_acc = 0
    best_kmeans_k = 0
    best_kmeans_pred = None
    
    for k in [2, 3, 4, 5, 6]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_clusters = kmeans.fit_predict(X_train_data)
        
        # Create cluster-to-label mapping
        cluster_labels = {}
        for cluster_id in range(k):
            cluster_mask = train_clusters == cluster_id
            cluster_true_labels = y_train_data[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
                cluster_labels[cluster_id] = most_common_label
        
        # Predict on test set
        test_clusters = kmeans.predict(X_test_data)
        y_pred_kmeans = np.array([cluster_labels.get(cluster_id, 0) for cluster_id in test_clusters])
        accuracy_kmeans = accuracy_score(y_test_data, y_pred_kmeans)
        print(f"  k={k}: {accuracy_kmeans:.4f}")
        
        if accuracy_kmeans > best_kmeans_acc:
            best_kmeans_acc = accuracy_kmeans
            best_kmeans_k = k
            best_kmeans_pred = y_pred_kmeans
    
    results['kmeans'] = {
        'best_k': best_kmeans_k,
        'predictions': best_kmeans_pred,
        'accuracy': best_kmeans_acc
    }
    
    print(f"\n{dataset_name} - Summary:")
    print(f"  Logistic Regression: {accuracy_lr:.4f}")
    print(f"  Best KNN (k={best_knn_k}): {best_knn_acc:.4f}")
    print(f"  Best K-Means (k={best_kmeans_k}): {best_kmeans_acc:.4f}")
    
    return results

# Run analysis on all four datasets
full_results = run_classification_analysis(X_train_scaled, X_test_scaled, y_train, y_test, "Full")
mean_results = run_classification_analysis(X_mean_train_scaled, X_mean_test_scaled, y_train, y_test, "Mean")
se_results = run_classification_analysis(X_se_train_scaled, X_se_test_scaled, y_train, y_test, "SE")
worst_results = run_classification_analysis(X_worst_train_scaled, X_worst_test_scaled, y_train, y_test, "Worst")

# Comprehensive comparison across all datasets
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON ACROSS DATASET DIVISIONS")
print("="*80)

# Create comparison tables
datasets = ['Full', 'Mean', 'SE', 'Worst']
all_results = [full_results, mean_results, se_results, worst_results]

# Accuracy comparison table
print("\nAccuracy Comparison Table:")
print(f"{'Dataset':<10} {'LR':<8} {'KNN':<8} {'K-Means':<8} {'Best Algorithm':<15}")
print("-" * 65)

best_overall = {'dataset': '', 'algorithm': '', 'accuracy': 0}

for dataset, results in zip(datasets, all_results):
    lr_acc = results['lr']['accuracy']
    knn_acc = results['knn']['accuracy']
    kmeans_acc = results['kmeans']['accuracy']
    
    # Find best algorithm for this dataset
    best_acc = max(lr_acc, knn_acc, kmeans_acc)
    if best_acc == lr_acc:
        best_alg = "Logistic Reg"
    elif best_acc == knn_acc:
        best_alg = f"KNN (k={results['knn']['best_k']})"
    else:
        best_alg = f"K-Means (k={results['kmeans']['best_k']})"
    
    print(f"{dataset:<10} {lr_acc:<8.4f} {knn_acc:<8.4f} {kmeans_acc:<8.4f} {best_alg:<15}")
    
    # Track overall best
    if best_acc > best_overall['accuracy']:
        best_overall = {'dataset': dataset, 'algorithm': best_alg, 'accuracy': best_acc}

print(f"\nOverall Best Performance: {best_overall['algorithm']} on {best_overall['dataset']} dataset ({best_overall['accuracy']:.4f})")

# Feature importance analysis
print("\n" + "="*60)
print("FEATURE CATEGORY IMPORTANCE ANALYSIS")
print("="*60)

# Calculate average performance for each feature category
mean_avg = np.mean([mean_results['lr']['accuracy'], mean_results['knn']['accuracy'], mean_results['kmeans']['accuracy']])
se_avg = np.mean([se_results['lr']['accuracy'], se_results['knn']['accuracy'], se_results['kmeans']['accuracy']])
worst_avg = np.mean([worst_results['lr']['accuracy'], worst_results['knn']['accuracy'], worst_results['kmeans']['accuracy']])

print(f"Average performance across all algorithms:")
print(f"  Mean features: {mean_avg:.4f}")
print(f"  SE features: {se_avg:.4f}")
print(f"  Worst features: {worst_avg:.4f}")

# Rank feature categories
category_performance = [('Mean', mean_avg), ('SE', se_avg), ('Worst', worst_avg)]
category_performance.sort(key=lambda x: x[1], reverse=True)

print(f"\nFeature category ranking:")
for i, (category, avg_acc) in enumerate(category_performance, 1):
    print(f"  {i}. {category} features: {avg_acc:.4f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy comparison by algorithm
algorithms = ['Logistic Regression', 'KNN', 'K-Means']
full_accs = [full_results['lr']['accuracy'], full_results['knn']['accuracy'], full_results['kmeans']['accuracy']]
mean_accs = [mean_results['lr']['accuracy'], mean_results['knn']['accuracy'], mean_results['kmeans']['accuracy']]
se_accs = [se_results['lr']['accuracy'], se_results['knn']['accuracy'], se_results['kmeans']['accuracy']]
worst_accs = [worst_results['lr']['accuracy'], worst_results['knn']['accuracy'], worst_results['kmeans']['accuracy']]

x = np.arange(len(algorithms))
width = 0.2

axes[0, 0].bar(x - 1.5*width, full_accs, width, label='Full Dataset', alpha=0.8)
axes[0, 0].bar(x - 0.5*width, mean_accs, width, label='Mean Features', alpha=0.8)
axes[0, 0].bar(x + 0.5*width, se_accs, width, label='SE Features', alpha=0.8)
axes[0, 0].bar(x + 1.5*width, worst_accs, width, label='Worst Features', alpha=0.8)

axes[0, 0].set_xlabel('Algorithm')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Performance Comparison Across Feature Categories')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(algorithms)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Feature category average performance
categories = ['Mean', 'SE', 'Worst']
avg_performances = [mean_avg, se_avg, worst_avg]
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = axes[0, 1].bar(categories, avg_performances, color=colors, alpha=0.8)
axes[0, 1].set_ylabel('Average Accuracy')
axes[0, 1].set_title('Average Performance by Feature Category')
axes[0, 1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, avg in zip(bars, avg_performances):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{avg:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Best performance for each dataset
best_performances = []
dataset_labels = []
for dataset, results in zip(datasets, all_results):
    lr_acc = results['lr']['accuracy']
    knn_acc = results['knn']['accuracy']
    kmeans_acc = results['kmeans']['accuracy']
    best_acc = max(lr_acc, knn_acc, kmeans_acc)
    best_performances.append(best_acc)
    dataset_labels.append(dataset)

bars = axes[1, 0].bar(dataset_labels, best_performances, color=['purple', 'orange', 'green', 'red'], alpha=0.8)
axes[1, 0].set_ylabel('Best Accuracy')
axes[1, 0].set_title('Best Performance per Dataset')
axes[1, 0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, best_performances):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Feature count and dimensionality
feature_counts = [X.shape[1], len(mean_features), len(se_features), len(worst_features)]
axes[1, 1].bar(dataset_labels, feature_counts, color=['gray', 'skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
axes[1, 1].set_ylabel('Number of Features')
axes[1, 1].set_title('Feature Count by Dataset')
axes[1, 1].grid(True, alpha=0.3)

# Add value labels on bars
for i, (label, count) in enumerate(zip(dataset_labels, feature_counts)):
    axes[1, 1].text(i, count + 0.5, f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('dataset_division_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis between feature categories
print("\n" + "="*60)
print("CORRELATION ANALYSIS BETWEEN FEATURE CATEGORIES")
print("="*60)

# Calculate correlations between feature category performances
from scipy.stats import pearsonr

# Create performance matrix
performance_matrix = np.array([
    [full_results['lr']['accuracy'], full_results['knn']['accuracy'], full_results['kmeans']['accuracy']],
    [mean_results['lr']['accuracy'], mean_results['knn']['accuracy'], mean_results['kmeans']['accuracy']],
    [se_results['lr']['accuracy'], se_results['knn']['accuracy'], se_results['kmeans']['accuracy']],
    [worst_results['lr']['accuracy'], worst_results['knn']['accuracy'], worst_results['kmeans']['accuracy']]
])

# Calculate pairwise correlations between datasets
dataset_names = ['Full', 'Mean', 'SE', 'Worst']
print("Performance correlation between datasets:")
for i in range(len(dataset_names)):
    for j in range(i+1, len(dataset_names)):
        corr, p_value = pearsonr(performance_matrix[i], performance_matrix[j])
        print(f"  {dataset_names[i]} vs {dataset_names[j]}: r = {corr:.4f} (p = {p_value:.4f})")

print("\nInterpretation:")
print("- High correlation suggests similar algorithm preferences")
print("- Low correlation suggests different optimal algorithms for different feature types")

# Create confusion matrices for the best performing combinations
print("\n" + "="*60)
print("CONFUSION MATRICES FOR BEST PERFORMERS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Find best performer for each dataset
datasets_info = [
    ('Full', full_results, X_train_scaled.shape[1]),
    ('Mean', mean_results, X_mean_train_scaled.shape[1]),
    ('SE', se_results, X_se_train_scaled.shape[1]),
    ('Worst', worst_results, X_worst_train_scaled.shape[1])
]

for idx, (dataset_name, results, n_features) in enumerate(datasets_info):
    # Find best algorithm for this dataset
    lr_acc = results['lr']['accuracy']
    knn_acc = results['knn']['accuracy'] 
    kmeans_acc = results['kmeans']['accuracy']
    
    best_acc = max(lr_acc, knn_acc, kmeans_acc)
    
    if best_acc == lr_acc:
        best_pred = results['lr']['predictions']
        title = f'{dataset_name} Dataset\nLogistic Regression ({best_acc:.4f})\n{n_features} features'
        cmap = 'Blues'
    elif best_acc == knn_acc:
        best_pred = results['knn']['predictions']
        title = f'{dataset_name} Dataset\nKNN k={results["knn"]["best_k"]} ({best_acc:.4f})\n{n_features} features'
        cmap = 'Greens'
    else:
        best_pred = results['kmeans']['predictions']
        title = f'{dataset_name} Dataset\nK-Means k={results["kmeans"]["best_k"]} ({best_acc:.4f})\n{n_features} features'
        cmap = 'Purples'
    
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx])
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('dataset_division_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Dataset with {X.shape[0]} samples and {X.shape[1]} original features")
print(f"Target classes: {y.value_counts().to_dict()}")
print(f"\nFeature division:")
print(f"  - Mean features: {len(mean_features)} ({len(mean_features)/len(all_features)*100:.1f}%)")
print(f"  - SE features: {len(se_features)} ({len(se_features)/len(all_features)*100:.1f}%)")  
print(f"  - Worst features: {len(worst_features)} ({len(worst_features)/len(all_features)*100:.1f}%)")

print(f"\nKey Findings:")
print(f"1. Best overall performance: {best_overall['algorithm']} on {best_overall['dataset']} dataset ({best_overall['accuracy']:.4f})")
print(f"2. Most informative feature category: {category_performance[0][0]} (avg accuracy: {category_performance[0][1]:.4f})")
print(f"3. Feature category ranking: {' > '.join([cat for cat, _ in category_performance])}")

# Performance improvement analysis
full_best = max(full_results['lr']['accuracy'], full_results['knn']['accuracy'], full_results['kmeans']['accuracy'])
category_best = max(mean_avg, se_avg, worst_avg)

if category_best > full_best:
    improvement = ((category_best - full_best) / full_best) * 100
    print(f"4. Using only {category_performance[0][0]} features improves performance by {improvement:.2f}% over full dataset")
else:
    degradation = ((full_best - category_best) / full_best) * 100
    print(f"4. Full dataset performs {degradation:.2f}% better than best single feature category")

print("\nThis analysis demonstrates the relative importance of different feature types")
print("in breast cancer diagnosis and can guide feature selection strategies.")

print("\n" + "="*80)
print("DATASET DIVISION ANALYSIS COMPLETE")
print("="*80)