import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTEENN
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #e63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; color: #6c757d; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SMOTEENN Resampling + XGBoost/Random Forest + SHAP Explainability</div>', unsafe_allow_html=True)

@st.cache_data
def generate_demo_data():
    """Generate realistic synthetic fraud detection data for demo"""
    np.random.seed(42)
    n_legit = 2000
    n_fraud = 100

    # Legitimate transactions
    legit = pd.DataFrame(
        np.random.randn(n_legit, 28),
        columns=[f'V{i}' for i in range(1, 29)]
    )
    legit['Amount'] = np.random.exponential(88, n_legit)
    legit['Time'] = np.sort(np.random.uniform(0, 172800, n_legit))
    legit['Class'] = 0

    # Fraudulent transactions (slightly different distribution)
    fraud = pd.DataFrame(
        np.random.randn(n_fraud, 28) * 1.5 + 0.5,
        columns=[f'V{i}' for i in range(1, 29)]
    )
    fraud['Amount'] = np.random.exponential(122, n_fraud)
    fraud['Time'] = np.random.uniform(0, 172800, n_fraud)
    fraud['Class'] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTEENN resampling
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return model, report, cm, roc_auc, X_test, y_test, X_resampled

# Sidebar
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Data Source", ["Demo Mode", "Upload CSV"])

if mode == "Demo Mode":
    st.sidebar.info("Using synthetic data that mirrors real credit card transaction patterns.")
    df = generate_demo_data()
    st.sidebar.success(f"Demo dataset: {len(df):,} transactions ({df['Class'].sum()} fraud cases)")
else:
    uploaded = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded: {len(df):,} rows")
    else:
        st.sidebar.warning("Upload a CSV with columns V1–V28, Amount, Time, Class")
        st.info("👈 Upload a CSV file or switch to Demo Mode to get started.")
        st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Results", "🔍 SHAP Explainability", "🔮 Predict Transaction"])

with tab1:
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    fraud_count = df['Class'].sum()
    legit_count = len(df) - fraud_count
    fraud_pct = fraud_count / len(df) * 100

    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Legitimate", f"{legit_count:,}")
    with col3:
        st.metric("Fraudulent", f"{fraud_count:,}")
    with col4:
        st.metric("Fraud Rate", f"{fraud_pct:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#2196F3', '#e63946']
        ax.bar(['Legitimate', 'Fraud'], [legit_count, fraud_count], color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Transaction Class Distribution')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Transaction Amount by Class")
        fig, ax = plt.subplots(figsize=(5, 4))
        df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6, label='Legitimate', color='#2196F3', ax=ax)
        df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6, label='Fraud', color='#e63946', ax=ax)
        ax.set_yscale('log')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Frequency (log)')
        ax.legend()
        ax.set_title('Amount Distribution')
        st.pyplot(fig)
        plt.close()

with tab2:
    st.header("Model Training & Results")
    st.info("Training Random Forest with SMOTEENN resampling to handle class imbalance...")

    with st.spinner("Training model..."):
        model, report, cm, roc_auc, X_test, y_test, X_resampled = train_model(df)

    st.success("✅ Model trained successfully!")

    col1, col2, col3, col4 = st.columns(4)
    fraud_report = report.get('1', report.get('Fraud', {}))
    with col1:
        st.metric("Accuracy", f"{report['accuracy']:.1%}")
    with col2:
        st.metric("Precision (Fraud)", f"{fraud_report.get('precision', 0):.1%}")
    with col3:
        st.metric("Recall (Fraud)", f"{fraud_report.get('recall', 0):.1%}")
    with col4:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Legit', 'Predicted Fraud'],
                    yticklabels=['Actual Legit', 'Actual Fraud'], ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Importance (Top 10)")
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('Top 10 Features')
        ax.set_xlabel('Importance')
        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("SHAP Explainability")
    st.markdown("SHAP values show **why** the model makes each prediction — making the model transparent and trustworthy.")

    with st.spinner("Calculating SHAP values (this takes ~30 seconds)..."):
        explainer = shap.TreeExplainer(model)
        sample = X_test.iloc[:100]
        shap_values = explainer.shap_values(sample)

    st.subheader("Feature Impact on Fraud Predictions")
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **How to read this:** Features at the top have the most influence on fraud detection.
    Higher SHAP values = stronger push toward classifying as fraud.
    """)

with tab4:
    st.header("Predict a Single Transaction")
    st.markdown("Enter transaction details to get an instant fraud probability score.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
        time = st.number_input("Time (seconds from first transaction)", min_value=0, value=50000)

    with col2:
        st.markdown("**PCA Features (V1-V4)** — key fraud indicators")
        v1 = st.slider("V1", -5.0, 5.0, 0.0)
        v2 = st.slider("V2", -5.0, 5.0, 0.0)
        v3 = st.slider("V3", -5.0, 5.0, 0.0)
        v4 = st.slider("V4", -5.0, 5.0, 0.0)

    if st.button("🔍 Analyze Transaction", type="primary"):
        input_data = pd.DataFrame([[time] + [v1, v2, v3, v4] + [0.0] * 24 + [amount]],
                                   columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
        # Reorder to match training columns
        input_data = input_data[X_test.columns]
        proba = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.error(f"⚠️ **FRAUD DETECTED** — Fraud Probability: {proba:.1%}")
        else:
            st.success(f"✅ **LEGITIMATE** — Fraud Probability: {proba:.1%}")

        st.progress(float(proba))
        st.caption(f"Risk Score: {proba:.3f} | Threshold: 0.5")

# Footer
st.markdown("---")
st.markdown("**Built by Ghanashyam T V** | [GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")
