# Customer Segmentation Analysis for E-commerce

## Project Description
This project performs customer segmentation for an e-commerce platform by clustering customers based on their demographic and purchasing behavior data. The goal is to identify distinct customer groups to enable targeted marketing strategies and improve customer engagement.

## 1. Project Objective
Segment customers of an e-commerce company into distinct groups based on purchasing behavior and demographics to enable targeted marketing strategies and improved customer engagement.

## 2. Dataset Information
- The dataset includes 39 features encompassing customer demographics (e.g., Income, Recency) and purchase activity (e.g., category spend, purchase frequency online and offline).
- Key features for clustering are Income, Recency, spending on wines, meats, sweets, gold products, and purchase counts on web and in-store.
- The original dataset is confidential and not included here. To run the code, replace `'ifood_df.csv'` in the script with a path to your dataset or a synthetically generated sample with similar columns.

## 3. Methodology
- **Preprocessing:** Selected 13 key behavioral features and applied standard scaling.  
- **Clustering:** Used K-Means algorithm with optimal cluster determination.  
- **Cluster Number Selection:**  Optimal number of clusters (2) chosen using Elbow Method and Silhouette Scores. 
- **Visualizations:**  Pairplots, heatmaps, box plots, and parallel coordinates plots for comprehensive cluster analysis.

## 4. Cluster Profiles
### Cluster 0 (High-Value Customers)
- Higher Income and significantly greater spending across all product categories.
- Higher frequency of purchases across web, catalog, and store channels.
- Likely highly engaged, premium customer segment.

### Cluster 1 (Low-Value Customers)
- Moderate to low income and spending across categories.
- Less frequent purchases and lower engagement across all channels.
- Represents budget-conscious or infrequent buyers.

Both clusters showed similar Recency distribution, indicating time since last purchase was not a primary differentiator.

### Key Features Used for Clustering:
- Income, Recency
- Spending: Wines, Fruits, Meat, Fish, Sweets, Gold products
- Purchase channels: Deals, Web, Catalog, Store purchases
- Web visits per month

### Algorithms Used:
- StandardScaler for feature normalization
- KMeans for clustering
- Silhouette Score and Elbow Method for optimal cluster determination

## 5. Business Implications

### Cluster 0 (High-Value Customers)
- **Target with premium strategies:** Loyalty programs, exclusive offers, and personalized recommendations
- **Maximize lifetime value:** Upsell complementary products and introduce subscription models
- **Retention focus:** Priority customer service and early access to new products

### Cluster 1 (Low-Value Customers)
- **Acquisition and activation:** Welcome discounts, first-purchase incentives, and educational content
- **Increase engagement:** Regular promotional campaigns and reminder emails for abandoned carts
- **Budget-friendly options:** Highlight value products and bundle deals

### Cross-Cluster Strategies
- **Personalized marketing:** Tailor messaging based on spending patterns and product preferences
- **Channel optimization:** Focus on preferred purchase channels for each segment
- **Product recommendations:** Suggest items based on cluster-specific buying behaviors

## 6. Key Insights from Customer Segmentation

### Spending Patterns
- **High-Value Cluster** shows significantly higher spending across all product categories, particularly wines and meat products
- **Low-Value Cluster** demonstrates moderate spending with more balanced distribution across categories

### Income and Behavior Correlation
- Higher income strongly correlates with increased spending across all product categories
- Income level is the primary differentiator between the two customer segments

### Purchase Channel Preferences
- Both clusters utilize multiple purchase channels (web, store, catalog)
- High-value customers show slightly higher engagement across all channels

### Recency Patterns
- Time since last purchase (Recency) is similar across both clusters
- Purchase recency is not a primary segmentation factor in this analysis

### Behavioral Segmentation
- The 2-cluster solution effectively separates customers based on overall engagement and spending capacity
- Clear distinction enables targeted marketing strategies

## 7. Hidden Relationships Discovered

### Product Category Interdependencies
- Customers who purchase wines are also likely to purchase meat products
- Sweet products and gold products show different purchasing patterns across clusters

### Channel Complementarity
- Web purchases and store purchases show positive correlation
- Customers use multiple channels rather than sticking to a single purchase method

### Income-Driven Behavior
- Income level is the strongest predictor of customer spending behavior
- Higher income enables broader product experimentation and larger purchases

### Purchase Frequency Patterns
- Number of purchases correlates with total spending amount
- Deal purchases are distributed differently across income segments

### Customer Lifetime Value Indicators
- Total spending (MntTotal) combined with purchase frequency serves as a strong CLV indicator
- High-value cluster demonstrates characteristics of loyal, repeat customers

## 8. Future Work
- Explore additional clustering algorithms such as hierarchical clustering or DBSCAN.  
- Expand feature set with customer lifetime value, product preferences, and seasonal effects.  
- Longitudinal analysis to track segment evolution over time.

## 9. Visualizations

Detailed visualization plots can be found in the [Visualizations document](visualizations.md).

---

## Installation Instructions
This project requires Python 3.x and the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies via pip:

pip install pandas numpy matplotlib seaborn scikit-learn


---

## How to Run This Project
1. Clone or download this repository.
2. Place your dataset CSV file in the project directory or specify the file path in the code.
3. Run the segmentation script or Jupyter notebook.
4. The script performs data preprocessing, clustering (K-Means), and creates visualizations such as:
    - Pairplots to explore feature relationships  
    - Elbow and silhouette plots for cluster evaluation  
    - Heatmaps and box plots for cluster profiling  
    - Parallel coordinates for multivariate cluster visualization  
5. View outputs in the console and saved plots as needed.

---

## Contact
For questions or collaboration, please reach out to me:

- **Name:** Ghanashyam T V  
- **Email:** [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)  
- **LinkedIn:** [www.linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring this project! Feel free to open issues or pull requests for improvements.

