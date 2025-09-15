# Project Visualizations

This document showcases key visualizations generated during the credit card fraud detection project. These plots help illustrate data characteristics, model performance, and actionable insights.

---

## 1. Transaction Class Distribution  
Displays counts and percentages of legitimate vs fraudulent transactions.  
**Insight:** Highlights the severe class imbalance (~0.17% fraud), critical for model design.  
![Transaction Class Counts](images/Figure_1.jpg)

---

## 2. Transaction Amount by Class  
Shows distribution and variation of transaction amounts for legitimate and fraudulent transactions.  
**Insight:** Fraudulent transactions tend to have smaller amounts but overlap considerably with legitimate ones, limiting predictive power of amount alone.  
![Transaction Amount by Class](images/Figure_2.jpg)

---

## 3. Confusion Matrix  
Visualizes the Random Forest model's classification results.  
**Insight:** Model achieves high precision but misses some fraud cases.  
![Confusion Matrix](images/Figure_3.jpg)

---

## 4. Top 5 Features Most Correlated with Fraud  
Bar chart showing the features with the strongest linear association with fraud.  
**Insight:** Highlights key PCA components driving fraud prediction.  
![Top Correlated Features](images/Figure_4.jpg)

---

## 5. ROC and Precision-Recall Curves  
Evaluates model discriminatory power (ROC AUC = 0.92) and performance on imbalanced data (Average Precision = 0.79).  
**Insight:** Demonstrates ability to identify fraud while balancing false positives.  
![ROC & PR Curves](images/Figure_5.jpg)

---

## 6. Top 10 Most Important Features  
Displays the features most influential to the Random Forest model predictions.  
**Insight:** Validates importance of PCA components and other engineered features for fraud detection.  
![Feature Importance](images/Figure_6.jpg)

---

*For further details, refer to the main project report and source code.*

