# Credit Card Fraud Detection Pipeline

## Project Description
This project implements a comprehensive credit card fraud detection system. It leverages advanced machine learning techniques to identify fraudulent transactions in imbalanced financial datasets. The goal is to maximize fraud detection accuracy while minimizing false alarms, enabling better fraud prevention and financial security.

## 1. Project Objective
Detect fraudulent credit card transactions using supervised machine learning models, with a focus on handling severe class imbalance and optimizing decision thresholds for real-world deployment.

## 2. Dataset Information
- The dataset contains 284,807 credit card transactions with 31 features including anonymized PCA components, transaction time, and amount.
- The class label is heavily imbalanced (~0.17% fraud cases).
- The original dataset (`creditcard.csv`) is not included due to size and licensing restrictions. Users should obtain it from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or supply their own dataset with the same schema.

## 3. Methodology
- **Exploratory Data Analysis (EDA):** Class distribution, transaction amounts, fraud timing patterns, pairplots, and correlation heatmaps.
- **Preprocessing:** Robust scaling of time and amount features.
- **Imbalance Handling:** Combination of SMOTE and Edited Nearest Neighbors (SMOTEENN) to balance training data.
- **Modeling:** Trained Logistic Regression, Random Forest, and GPU-accelerated XGBoost classifiers.
- **Evaluation:** Metrics include Precision, Recall, F1-score, ROC AUC, and Average Precision. Visualized confusion matrices, ROC, and precision-recall curves.
- **Threshold Tuning:** Identified optimal classification decision threshold based on maximizing F1-score.
- **Explainability:** SHAP-based feature importance and summary plots to interpret model predictions.
- **Final Recommendations:** Identified the best model and provided actionable business insights.

## 4. Key Results
- **Random Forest** outperformed other models, achieving:
  - F1-score of 0.73 (default threshold)
  - ROC AUC of 0.98
  - Optimized threshold at 0.90 with Precision 90% and Recall 74%
- Important fraud indicators were found in features V3, V12, and V14, as well as unusual transaction amounts and timing.

## 5. Business Implications
- Deploying the Random Forest model with tuned threshold will detect the majority of frauds while significantly reducing false positives.
- Features identified by SHAP can guide further investigations and validation by fraud analysts.
- The pipeline supports data-driven decisions to improve fraud prevention and customer trust.

## 6. Future Work
- Incorporate additional feature engineering using transactional and customer metadata.
- Experiment with ensemble methods and deep learning-based anomaly detection.
- Develop real-time scoring API for deployment.
- Implement continuous learning and retraining pipelines to adapt to evolving fraud patterns.

## 7. Visualizations

Detailed project visualizations can be found in the [VISUALIZATIONS.md](./VISUALIZATIONS.md) file.

Alternatively, view the `images/` directory for individual plot images.

---

## Installation Instructions

This project requires Python 3.x and the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- shap

Install dependencies via pip:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap



---

## How to Run This Project

1. Clone or download this repository.
2. Place `creditcard.csv` in the project root directory or update the file path in the code.
3. Activate your virtual environment (recommended).
4. Run the pipeline script:

python fraud_detection.py


5. View results and plots generated during execution.

---


## Contact

For questions or collaboration, please reach out:

- **Name:** Ghanashyam T V  
- **Email:** [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)  
- **LinkedIn:** [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring this project! Feel free to open issues or pull requests 