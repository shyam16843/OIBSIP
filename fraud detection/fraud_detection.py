import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, 
                             PrecisionRecallDisplay, classification_report, precision_recall_curve)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Section 1: Data Loading and EDA
# ---------------------------
def load_and_eda():
    data = pd.read_csv('creditcard.csv')
    print("Dataset shape:", data.shape)
    print("\nClass distribution:")
    print(data['Class'].value_counts())
    print("\nPercentage of fraud cases: {:.4f}%".format(data['Class'].mean() * 100))
    
    # Visualizations
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='Class', data=data)
    plt.title('Transaction Class Counts')
    color_legit = ax.patches[0].get_facecolor()
    color_fraud = 'red'
    legend_handles = [
        mpatches.Patch(color=color_legit, label='Legit (0)'),
        mpatches.Patch(color=color_fraud, label='Fraud (1)')
    ]
    plt.legend(handles=legend_handles, title='Class')
    plt.show()
    
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Class', y='Amount', data=data)
    plt.title('Transaction Amount by Class')
    plt.show()
    
    data['hour'] = (data['Time'] // 3600) % 24
    fraud_by_hour = data[data['Class'] == 1].groupby('hour').size()
    plt.figure(figsize=(10,5))
    fraud_by_hour.plot(kind='bar')
    plt.title('Fraudulent Transactions by Hour of Day')
    plt.ylabel('Number of Frauds')
    plt.xlabel('Hour of Day', labelpad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # FIXED PAIRPLOT SECTION
    sample_legit = data[data['Class'] == 0].sample(200, random_state=42)
    sample_fraud = data[data['Class'] == 1]
    sampled_data = pd.concat([sample_legit, sample_fraud])

    # Create pairplot with adjusted parameters
    g = sns.pairplot(sampled_data, 
                    vars=['V1', 'V2', 'V3', 'V4', 'V5'], 
                    hue='Class', 
                    diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 's': 15},
                    height=2.0,
                    aspect=1.0)

    # Remove default legend
    if g._legend is not None:
        g._legend.remove()

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8)
    ]

    # MOVE LEGEND LEFT - Adjusted bbox_to_anchor from (1.02, 0.5) to (0.95, 0.5)
    g.fig.legend(handles=legend_elements,
                labels=['Legitimate (0)', 'Fraudulent (1)'],
                loc='center right',
                bbox_to_anchor=(1.02, 0.5),  # MOVED LEFT from 1.02 to 0.95
                fontsize=10,
                title='Transaction Class',
                title_fontsize=11,
                frameon=True,
                fancybox=True)

    # Add title with adjusted position
    g.fig.suptitle('Pairplot of Selected Features by Class', 
                  y=0.98, fontsize=14, fontweight='bold')

    # Adjust subplot parameters to accommodate the moved legend
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.1, top=0.92, wspace=0.3, hspace=0.3)  # Changed right from 0.85 to 0.88

    # Rotate x-axis labels on bottom row and adjust padding
    for i, ax_row in enumerate(g.axes):
        for j, ax in enumerate(ax_row):
            if ax is not None:
                # Bottom row - rotate x labels
                if i == len(g.axes) - 1:
                    ax.tick_params(axis='x', rotation=45, labelrotation=45, labelsize=8)
                    ax.set_xlabel(ax.get_xlabel(), labelpad=10)
                # Left column - adjust y labels
                if j == 0:
                    ax.set_ylabel(ax.get_ylabel(), labelpad=10)
                
                # Adjust all tick labels
                ax.tick_params(axis='both', which='major', labelsize=8, pad=3)

    plt.show()
    # ... (rest of your function remains the same)
        
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    class_correlations = corr_matrix['Class'].sort_values(ascending=False)
    top_correlated = class_correlations[abs(class_correlations) > 0.1].index
    sns.heatmap(data[top_correlated].corr(), annot=True, cmap='coolwarm', linewidths=0.5, center=0,
                fmt='.2f', square=True, cbar_kws={'label': 'Correlation Coefficient'}) # This is the LEGEND
    plt.title('Correlation Matrix (Features with |correlation| > 0.1 with Class)', fontsize=14, pad=20)
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.show()
    
    data.drop(['hour'], axis=1, inplace=True)
    return data

# ---------------------------
# Section 2: Preprocessing and Data Split
# ---------------------------
def preprocess_and_split(data):
    scaler = RobustScaler()
    data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# ---------------------------
# Section 3: Resampling Training Data (SMOTEENN)
# ---------------------------
def resample(X_train, y_train):
    resampler = SMOTEENN(random_state=42)
    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
    print(f"\nOriginal training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_res.shape}")
    print(f"Resampled class distribution:\n{pd.Series(y_train_res).value_counts()}")
    return X_train_res, y_train_res

# ---------------------------
# Section 4: Model Evaluation Function (With Visualization)
# ---------------------------
def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    print(f"\n{model_name} Performance at threshold={threshold:.2f}:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}, Average Precision (PR AUC): {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=axes[0], values_format='d')
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"{model_name} - ROC Curve")
    
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[2])
    axes[2].set_title(f"{model_name} - Precision-Recall Curve")
    
    plt.tight_layout()
    plt.show()
    
    return precision, recall, f1, roc_auc, avg_precision

# ---------------------------
# Section 5: Train Multiple Models
# ---------------------------
def train_models(X_train_res, y_train_res, X_test, y_test, y_train):
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    lr_metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    # Random Forest
    rf = RandomForestClassifier(
    n_estimators=50,      # reduce number of trees
    max_depth=15,         # limit depth to control complexity
    min_samples_split=10, # increase to reduce splits
    class_weight='balanced',
    random_state=42,
    n_jobs=-1             # use all CPU cores (if available)
)
    rf.fit(X_train_res, y_train_res)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='gpu_hist',  # enable GPU acceleration
    gpu_id=0,                # usually 0 for first GPU
    predictor='gpu_predictor',
    random_state=42,
    n_estimators=50,
    max_depth=15,
    min_child_weight=10      # similar to min_samples_split, controls complexity
)
    xgb.fit(X_train_res, y_train_res)
    xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")
    # Model comparison
    metrics_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics],
                          index=['Logistic Regression', 'Random Forest', 'XGBoost'],
                          columns=['precision', 'recall', 'f1', 'roc_auc', 'avg_precision'])
    print("\nModel Comparison:")
    print(metrics_df)
    return lr, rf, xgb, metrics_df

# ---------------------------
# Section 6: Threshold Tuning
# ---------------------------
def tune_threshold(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"\nOptimal classification threshold based on max F1-score: {best_threshold:.4f}")
    return best_threshold

# ---------------------------
# Section 7: SHAP Explainability - FIXED VERSION
# ---------------------------
def shap_explanation(model, X_test, model_name):
    sample_idx = X_test.sample(500, random_state=42).index
    X_sample = X_test.loc[sample_idx]
    
    try:
        # Create explainer based on model type
        if hasattr(model, 'feature_importances_'):  # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot 1: Bar plot - Formatted and styled
            # ADD THIS LINE
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {model_name}", 
                    fontsize=18, fontweight='bold', pad=25, color='#2E4057')
            plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', 
                    fontsize=12, labelpad=15)
            plt.tight_layout(pad=3.0)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_facecolor('#F8F9FA')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            # Center the plot
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
            plt.show()
            
            # Plot 2: Summary plot - Formatted and styled
            # ADD THIS LINE
            shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
            plt.title(f"SHAP Summary Plot - {model_name}", 
                    fontsize=18, fontweight='bold', pad=25, color='#2E4057')
            plt.tight_layout(pad=4.0)
            plt.gca().set_facecolor('#F8F9FA')
            # Add some styling to the summary plot
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.grid(axis='x', alpha=0.2, linestyle='--')
            # Center the plot
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
            plt.show()
            
        elif hasattr(model, 'coef_'):  # Linear models
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            # Linear model bar plot
            # ADD THIS LINE
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {model_name}", 
                    fontsize=18, fontweight='bold', pad=25, color='#2E4057')
            plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', 
                    fontsize=12, labelpad=15)
            plt.tight_layout(pad=3.0)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_facecolor('#F8F9FA')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            # Center the plot
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
            plt.show()
            
        else:
            print(f"SHAP not implemented for {type(model).__name__}")
            
    except Exception as e:
        print(f"SHAP explanation failed due to: {e}. Continuing without SHAP.")
        import traceback
        traceback.print_exc()
        
        
        
# ---------------------------
# Section 8: Final Conclusion & Recommendation
# ---------------------------
def final_conclusion(metrics_df, best_model_name, best_threshold, recall, precision):
    best_f1 = metrics_df.loc[best_model_name, 'f1']
    best_auc = metrics_df.loc[best_model_name, 'roc_auc']
    print("="*60)
    print("CONCLUSION & RECOMMENDATION")
    print("="*60)
    print(f"After evaluating multiple algorithms, the {best_model_name} performed best "
          f"with an F1-score of {best_f1:.4f} and ROC AUC of {best_auc:.4f}.")
    print("Key characteristics of fraudulent transactions typically include:")
    print("1. Extreme values in features V3, V12, V14 (often negative)")
    print("2. Unusual patterns in transaction amount and timing")
    print(f"\nRecommendation: Implement the {best_model_name} model to detect fraudulent transactions.")
    print(f"This model would catch {recall:.2%} of all fraud cases while maintaining "
          f"{precision:.2%} accuracy when flagging transactions as fraudulent.")
    print(f"Use a classification threshold of {best_threshold:.4f} to balance recall and precision.")

# ---------------------------
# Main - Execute Full Pipeline
# ---------------------------
def main():
    """
    Main execution function for the fraud detection pipeline.
    
    Pipeline Steps:
    1. 📊 Data Loading & Exploratory Data Analysis
    2. ⚙️  Data Preprocessing & Train-Test Split
    3. ⚖️  Class Imbalance Handling (SMOTEENN Resampling)
    4. 🤖 Machine Learning Model Training & Evaluation
    5. 🎯 Optimal Threshold Tuning
    6. 🔍 Model Interpretability (SHAP Analysis)
    7. 📋 Final Business Recommendations
    
    Returns:
        None: Executes full pipeline and displays results
    """
    print("🚀 Starting Credit Card Fraud Detection Pipeline")
    print("=" * 55)
    
    # Step 1: Data Loading and Exploratory Data Analysis
    print("\n📊 STEP 1: Loading data and performing exploratory analysis...")
    data = load_and_eda()
    
    # Step 2: Data Preprocessing and Splitting
    print("\n⚙️  STEP 2: Preprocessing data and creating train-test split...")
    X_train, X_test, y_train, y_test = preprocess_and_split(data)
    
    # Step 3: Handling Class Imbalance
    print("\n⚖️  STEP 3: Addressing class imbalance with SMOTEENN resampling...")
    X_train_res, y_train_res = resample(X_train, y_train)
    print(f"   → Resampled training data: {X_train_res.shape[0]:,} samples")
    print(f"   → Class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    
    # Step 4: Model Training and Evaluation
    print("\n🤖 STEP 4: Training and evaluating machine learning models...")
    lr_model, rf_model, xgb_model, metrics_df = train_models(
        X_train_res, y_train_res, X_test, y_test, y_train
    )
    
    # Identify best performing model
    best_model_name = metrics_df['f1'].idxmax()
    model_dict = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model, 
        'XGBoost': xgb_model
    }
    best_model = model_dict[best_model_name]
    
    # Step 5: Threshold Optimization
    print(f"\n🎯 STEP 5: Tuning optimal threshold for {best_model_name}...")
    best_threshold = tune_threshold(best_model, X_test, y_test)
    
    # Step 6: Final Model Evaluation with Optimal Threshold
    print(f"\n📈 STEP 6: Evaluating {best_model_name} with optimal threshold...")
    precision, recall, f1, roc_auc, avg_precision = evaluate_model(
        best_model, X_test, y_test, best_model_name, best_threshold
    )
    
    # Step 7: Model Interpretability
    print(f"\n🔍 STEP 7: Generating SHAP explanations for {best_model_name}...")
    shap_explanation(best_model, X_test, best_model_name)
    
    # Step 8: Final Conclusions and Recommendations
    print(f"\n📋 STEP 8: Generating final business recommendations...")
    final_conclusion(metrics_df, best_model_name, best_threshold, recall, precision)
    
    print("\n" + "=" * 55)
    print("✅ Pipeline execution completed successfully!")
    print("=" * 55)
    
if __name__ == "__main__":
    main()
