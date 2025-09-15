"""
Customer Segmentation Analysis for E-commerce
Goal: Identify customer segments based on purchasing behavior to inform targeted marketing.
Techniques: Data preprocessing, exploratory data analysis, clustering with KMeans,
and visualization of cluster profiles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas.plotting import parallel_coordinates

def load_data(path):
    # Load dataset and show basic information
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    print("Sample data:")
    print(df.head())
    return df

def preprocess_features(df):
    # Select relevant features for clustering (customer behavior & spending)
    features = [
        'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    X = df[features]

    # Scale features to zero mean and unit variance to ensure fair clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")
    return X_scaled, features

def exploratory_plots(df):
    # Visualize distributions and relationships between key features with pairplot
    selected_features = ['Income', 'Recency', 'MntWines', 'NumWebPurchases', 'NumStorePurchases']
    sample_df = df.sample(200, random_state=42)  # Sampling to reduce plot complexity
    
    # Create compact pairplot with smaller markers and smaller height per subplot
    g = sns.pairplot(
        sample_df[selected_features],
        diag_kind='kde',
        kind='scatter',
        plot_kws={'alpha': 0.6, 's': 20},  # smaller marker size for less clutter
        height=2.0,                        # smaller height per subplot for compact size
        aspect=1.2                     # keep square plots
    )
    
    # Set a clear title with little vertical padding
    g.fig.suptitle('Pairplot of Selected Customer Features', y=0.95, fontsize=14)
    
    # Rotate ONLY y-axis labels (all subplots)
    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(axis='y', rotation=90)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def find_optimal_clusters(X_scaled):
    # Use Elbow and Silhouette methods to find best number of clusters
    wcss = []  # Within cluster sum of squares
    silhouette_scores = []  # Silhouette coefficients
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Plot Elbow method (WCSS vs. number of clusters)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, wcss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within Cluster Sum of Squares')
    plt.title('Elbow Method')

    # Plot Silhouette scores vs number of clusters
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o')
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    plt.axvline(best_k, linestyle='--', color='red', label=f'Best k = {best_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Method')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Optimal number of clusters determined: {best_k}")
    return best_k

def perform_clustering(X_scaled, n_clusters):
    # Apply KMeans clustering with optimal clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    print(f"Clustering completed with {n_clusters} clusters.")
    return clusters

def plot_cluster_profiles(df, features, cluster_labels):
    # Attach cluster assignments and plot heatmap of average feature values by cluster
    df['Cluster'] = cluster_labels
    profile = df.groupby('Cluster')[features].mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(profile.T, annot=True, cmap='YlGnBu')
    plt.title('Cluster Profiles: Average Feature Values')
    plt.ylabel('Features')
    plt.xlabel('Cluster')
    # Move plot to the left by adjusting the right margin
    plt.subplots_adjust(right=1.1)  # Default is ~0.9, smaller = more left
    plt.show()

def plot_parallel_coordinates(df, features):
    # Plot parallel coordinates to visualize feature trends across clusters
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df[['Cluster'] + features], 'Cluster', colormap='Set2')
    plt.title('Parallel Coordinates Plot by Cluster')
    plt.xticks(rotation=30)
    plt.subplots_adjust(bottom=0.15)  # Default is ~0.1, larger value = moves plot up
    plt.show()

def plot_feature_boxplots(df):
    # Show distribution of key features per cluster via boxplots for outlier detection and spread
    key_features = ['Income', 'Recency', 'MntWines', 'NumWebPurchases']
    for feat in key_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Cluster', y=feat, hue='Cluster', palette='pastel', legend=False)
        plt.title(f'Distribution of {feat} Across Clusters')
        plt.show()

def main():
    # Overall workflow for customer segmentation analysis
    
    # Load and inspect data
    df = load_data('ifood_df.csv')
    
    # Prepare features
    X_scaled, features = preprocess_features(df)
    
    # Exploratory Data Analysis
    exploratory_plots(df)
    
    # Find best cluster count
    optimal_k = find_optimal_clusters(X_scaled)
    
    # Cluster customers
    clusters = perform_clustering(X_scaled, optimal_k)
    
    # Add cluster labels to dataframe
    df['Cluster'] = clusters
    
    # Visualize cluster profiles
    plot_cluster_profiles(df, features, clusters)
    
    # Visualize clusters with parallel coordinates plot
    plot_parallel_coordinates(df, features)
    
    # Boxplots for key features across clusters to understand distributions
    plot_feature_boxplots(df)
    
    print("Customer segmentation analysis is complete.")

if __name__ == "__main__":
    main()
