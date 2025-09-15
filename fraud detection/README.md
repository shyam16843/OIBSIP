# Credit Card Fraud Detection Pipeline

## Project Description
This project develops a credit card fraud detection pipeline to help identify and prevent fraudulent transactions. It focuses on handling severe class imbalance, exploratory data analysis, basic modeling, and delivering actionable business insights to enhance fraud prevention and financial security.

## 1. Project Objective
Detect fraudulent credit card transactions using supervised machine learning models, emphasizing data preprocessing, class imbalance handling, and optimized decision thresholds for real-world applicability.

## 2. Dataset Information
- The dataset contains 284,807 credit card transactions with 31 features including anonymized PCA components, transaction time, and amount.
- The class label is highly imbalanced, with only ~0.17% of transactions labeled as fraud, posing significant detection challenges.
- The original dataset (creditcard.csv) is not included due to size and licensing restrictions. Users should obtain it from Kaggle Credit Card Fraud Detection or supply their own dataset with the same schema.

## 3. Methodology
- **Exploratory Data Analysis (EDA):** Examination of class distribution, transaction amounts, fraud timing patterns, pairplots, and correlation heatmaps to understand data characteristics.
- **Preprocessing:**  Applied Robust scaling to handle outliers in transaction time and amount.
- **Imbalance Handling:** Employed SMOTEENN, combining synthetic minority sample generation and noise removal, to balance the training data.
- **Modeling:**Trained Logistic Regression, Random Forest, and GPU-accelerated XGBoost classifiers to evaluate predictive performance.
- **Evaluation:** Used metrics including Precision, Recall, F1-score, ROC AUC, and Average Precision. Visualized confusion matrices, ROC, and precision-recall curves to assess model quality.
- **Threshold Tuning:** Identified optimal classification thresholds by maximizing F1-score to balance fraud detection and false alarms.
- **Explainability:** Generated SHAP-based feature importance and summary plots for model interpretability.
- **Final Recommendations:** Selected the best-performing model and provided actionable business insights for fraud prevention.

## 4. Key Results
- **Random Forest** 
| Model         | F1-Score | ROC AUC | Precision (Optimized) | Recall (Optimized) |
| ------------- | -------- | ------- | -------------------- | ------------------ |
| Random Forest | 0.73     | 0.92    | 90%                  | 74%                |


The model effectively detects high-confidence fraud cases while reducing false positives.

Key fraud indicators include features V3, V12, V14, along with transaction amount patterns.

## 5. Business Implications
- High precision minimizes false alarms, reducing investigation costs and operational workload.
- Moderate recall highlights potential frauds missed; further model tuning or advanced methods are advised.
- The model is suitable as an initial filter prioritizing transactions for manual review.
- Important features identified can guide ongoing fraud analytics and feature engineering efforts.

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

Prerequisites
Python 3.8 or later recommended.

Environment Setup
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows

Install Dependencies
Install required Python packages using:

bash
pip install -r requirements.txt

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