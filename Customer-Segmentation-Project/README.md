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
- **Preprocessing:** Selected key behavioral features and applied standard scaling.  
- **Clustering:** Used K-Means algorithm.  
- **Cluster Number Selection:** Optimal number of clusters chosen using Elbow Method and Silhouette Scores, resulting in two distinct clusters.  
- **Visualizations:** Pairplots, heatmaps, box plots, and parallel coordinates plots for cluster profiling and validation.

## 4. Cluster Profiles
### Cluster 0 (High-Value Customers)
- Higher Income and significantly greater spending on key products (wines, meat, etc.).  
- Higher frequency of both online and in-store purchases.  
- Likely highly engaged, premium customer segment.

### Cluster 1 (Low-Value Customers)
- Moderate to low income and spending.  
- Less frequent purchases and lower engagement.  
- Represents budget-conscious or infrequent buyers.

Both clusters showed similar Recency distribution, indicating time since last purchase was not a primary differentiator.

## 5. Business Implications
- **High-Value Customers:** Target with loyalty rewards, premium offers, and upsell campaigns to maximize lifetime value.  
- **Low-Value Customers:** Focus on conversion strategies, discounts, and engagement tactics to increase purchase frequency and retention.

## 6. Overall Insights Uncovered
- **Dominant Product Categories:** Liquor, Wine, and Beer lead total retail sales with strong seasonal patterns indicating clear consumer demand cycles.  
- **Top Revenue Contributors:** A small number of key suppliers contribute the majority of sales, suggesting focused supplier management opportunities.  
- **Seasonality and Demand Cycles:** Spikes in sales in January and July align with promotional or holiday periods, informing inventory and marketing planning.  
- **Channel Interdependencies:** Correlation between sales channels reveals operational flow connections that can optimize supply chains.  
- **Sales Variability and Outliers:** Significant variability highlights potential bulk purchases or data anomalies needing further analysis.  
- **Trend Clarification:** Moving average smoothing aids in identifying reliable demand trends.  
- **Anomaly Detection:** Flags unusual spikes signaling key business events or risks.

## 7. Hidden Connections Revealed
- Seasonal effects vary by product line and supplier, guiding tailored marketing and stocking policies.  
- Sales channel correlations indicate logistic coordination potential.  
- Diverse supplier behaviors identified via sales distribution plots.  
- Anomaly detection segments normal vs exceptional sales cases.  
- Temporal sales shifts indicate changing customer preferences.

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

