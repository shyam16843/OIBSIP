# Customer Segmentation Project 

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates

# Step 1: Load Dataset
df = pd.read_csv('ifood_df.csv')  # Adjust the path if needed
print("Dataset loaded successfully")
print(df.head())

# Step 2: Feature Selection and Scaling
# Select key customer behavior features
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

X = df[features]

# Scale features to standard normal for clustering algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled")

# Step 3: Visualize feature relationships (Pairplot)
selected_features = ['Income', 'Recency', 'MntWines', 'NumWebPurchases', 'NumStorePurchases']
sns.pairplot(df[selected_features].sample(200), diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Step 4: Determine optimal number of clusters with Elbow and Silhouette methods
wcss = []
silhouette_avg = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

best_k = K_range[silhouette_avg.index(max(silhouette_avg))]
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_avg, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
plt.legend()
plt.tight_layout()
plt.show()

# Choose optimal clusters (e.g., from above)
k_optimal = best_k

# Step 5: Apply KMeans clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print("Cluster counts:")
print(df['Cluster'].value_counts())

# Step 6: Cluster Profiles Heatmap
cluster_profile = df.groupby('Cluster')[features].mean()
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile.T, annot=True, cmap='YlGnBu')
plt.title('Cluster Profiles Heatmap')
plt.show()

# Step 7: Parallel Coordinates Plot to view clusters
plt.figure(figsize=(12, 6))
parallel_coordinates(df[['Cluster'] + features], class_column='Cluster', colormap='Set2')
plt.title('Parallel Coordinates Plot by Cluster')
plt.xticks(rotation=45)
plt.show()

# Step 8: Box Plots for key features by cluster
key_features = ['Income', 'Recency', 'MntWines', 'NumWebPurchases']
for feature in key_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y=feature, data=df, color='skyblue')
    plt.title(f'Box Plot of {feature} by Cluster')
    plt.show()

# Step 9: Outlier Detection using boxplots on raw data (Income, Recency)
for feature in ['Income', 'Recency']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[feature], color='lightcoral')
    plt.title(f'Outlier Detection for {feature}')
    plt.show()
