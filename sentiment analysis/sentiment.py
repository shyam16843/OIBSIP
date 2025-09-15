# sentiment_analysis.py
# Twitter Sentiment Analysis
# Data Exploration and Model Training

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import sys
import time
import warnings

# Suppress matplotlib warnings safely
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib')

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD

# Download required NLTK datasets
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Set plotting styles
plt.style.use('ggplot')
sns.set_palette("husl")

def load_data(file_path, sample_size=None):
    """Load data, with optional sampling to speed up development."""
    print("Loading data...")
    df = pd.read_csv(file_path)
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} records for experimentation...")
        df = df.sample(sample_size, random_state=42)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    """Remove rows missing critical data."""
    print("Cleaning data...")
    initial_count = df.shape[0]
    df = df.dropna(subset=['clean_text', 'category'])
    print(f"Removed {initial_count - df.shape[0]} rows with missing values.")
    print(f"Data shape after cleaning: {df.shape}")
    return df

def preprocess_text_data(df):
    """
    Clean raw text data by:
    - Lowercasing
    - URL and special character removal
    - Tokenizing and lemmatizing
    - Stopword removal
    """
    print("Preprocessing text...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return ' '.join(words)
    df['cleaned_text'] = df['clean_text'].astype(str).apply(preprocess_text)
    df['text_length'] = df['cleaned_text'].apply(len)
    return df

def generate_eda_plots(df):
    """Generate exploratory data visualizations - separate plots."""
    print("Generating exploratory data visualizations...")

    # Sentiment distribution bar plot
    plt.figure(figsize=(8,6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#4CAF50', '#FFC107', '#F44336']
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Text length distribution per sentiment
    plt.figure(figsize=(10,6))
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        subset = df[df['sentiment'] == sentiment]
        sns.histplot(subset['text_length'], label=sentiment, kde=True, bins=30, alpha=0.6)
    plt.title('Text Length Distribution by Sentiment')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Sentiment distribution pie chart
    plt.figure(figsize=(8,8))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Distribution (%)')
    plt.tight_layout()
    plt.show()

def generate_wordclouds(df):
    """Generate word clouds separately for each sentiment."""
    print("Generating word clouds...")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white',
                           colormap='viridis', max_words=50).generate(text)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {sentiment}')
            plt.tight_layout()
            plt.show()

def extract_features(df, max_features=5000, use_dimensionality_reduction=False):
    """Extract TF-IDF features, optionally apply SVD for dimensionality reduction."""
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['category'].astype(int)
    print(f"Feature matrix shape: {X.shape}")
    if use_dimensionality_reduction and X.shape[0] > 10000:
        print("Applying Truncated SVD.")
        svd = TruncatedSVD(n_components=100, random_state=42)
        X = svd.fit_transform(X)
        print(f"Reduced feature matrix shape: {X.shape}")
    return X, y, tfidf

def train_model(X_train, y_train, use_fast_svm=True, n_jobs=-1):
    """Train multiple models and compare their performance."""
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
    }
    if use_fast_svm:
        # SGDClassifier approximates SVM efficiently on big datasets using stochastic gradient
        models['SVM (SGD)'] = SGDClassifier(loss='hinge', max_iter=1000, random_state=42, n_jobs=n_jobs)
    else:
        models['SVM (Linear)'] = LinearSVC(max_iter=1000, random_state=42, dual=False)

    results = {}
    times = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        start = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=n_jobs)
        duration = time.time() - start
        results[name] = scores
        times[name] = duration
        print(f"{name}: Mean F1={scores.mean():.4f} (+/- {scores.std():.4f}), Time={duration:.2f}s\n")

    # Plot accuracy
    plt.figure(figsize=(10,6))
    bars = plt.bar(results.keys(), [np.mean(x) for x in results.values()],
                   color=['#4CAF50', '#FFC107', '#03A9F4', '#FF5722'])
    plt.title('Model Accuracy (Weighted F1 Score)')
    plt.ylabel('F1 Score')
    plt.ylim(0.6, 0.9)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha='center')
    plt.tight_layout()
    plt.show()

    # Plot training time
    plt.figure(figsize=(10,6))
    bars = plt.bar(times.keys(), list(times.values()),
                   color=['#4CAF50', '#FFC107', '#03A9F4', '#FF5722'])
    plt.title('Model Training Time (seconds)')
    plt.ylabel('Seconds')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}s", ha='center')
    plt.tight_layout()
    plt.show()

    return results, times

def plot_multiclass_roc_pr_curves(y_test, y_score, class_names):
    """Plot ROC and Precision-Recall curves per class."""
    y_test_bin = label_binarize(y_test, classes=sorted(class_names, key=lambda x: {'Negative': -1, 'Neutral': 0, 'Positive': 1}[x]))
    n_classes = y_test_bin.shape[1]

    # ROC Curves
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curves
    precision, recall, pr_auc = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(10, 8))
    for i, color in enumerate(colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_names[i]} (AP = {pr_auc[i]:.2f})')
    plt.title('Multi-class Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model with classification report, confusion matrix and plots."""
    import numpy as np
    
    print("Evaluating model...")
    start = time.time()
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None
    duration = time.time() - start

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f} ; Prediction time: {duration:.4f} seconds")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    if y_proba is not None:
        # Ensure alignment of classes between y_test and model output
        model_classes = model.classes_
        print(f"Model classes: {model_classes}")
        print(f"Unique labels in y_test: {np.unique(y_test)}")

        # Create binary indicator matrix, aligning labels properly
        y_test_bin = label_binarize(y_test, classes=model_classes)
        print(f"y_test_bin shape: {y_test_bin.shape}, y_proba shape: {y_proba.shape}")

        plot_multiclass_roc_pr_curves(y_test_bin, y_proba, [str(c) for c in model_classes])

    return y_pred, y_proba, accuracy_score(y_test, y_pred), duration


def plot_multiclass_roc_pr_curves(y_test_bin, y_score, class_names):
    import numpy as np
    # Number of classes
    n_classes = y_test_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, pr_auc = {}, {}, {}

    for i in range(n_classes):
        # Check for presence of positive samples in this class
        if np.sum(y_test_bin[:, i]) == 0:
            print(f"Warning: No positive samples for class {class_names[i]}. Skipping ROC and PR for this class.")
            # Plot flat lines to indicate no info
            fpr[i], tpr[i], roc_auc[i] = [0, 1], [0, 1], float('nan')
            precision[i], recall[i], pr_auc[i] = [1, 0], [0, 1], 0.
        else:
            # Compute ROC curve and AUC
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute Precision-Recall curve and AUC
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})" if not np.isnan(roc_auc[i]) else f"{class_names[i]} (AUC = N/A)")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Multi-class ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], color=colors[i], lw=2,
                 label=f"{class_names[i]} (AP = {pr_auc[i]:.2f})" if pr_auc[i] != 0. else f"{class_names[i]} (AP = N/A)")
    plt.title("Multi-class Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def optimize_sample(df, X, target, sample_size=10000):
    """If dataset is too large, returns a smaller random sample."""
    if len(df) > sample_size:
        print(f"Sampling {sample_size} records for faster training.")
        idx = np.random.choice(len(df), sample_size, replace=False)
        return X[idx], target.iloc[idx]
    return X, target

def main():
    print("Starting sentiment analysis...")
    start_time = time.time()

    SAMPLE_SIZE = None  # e.g. 10000 for quick testing; None for full dataset
    USE_FAST_SVM = True
    DIM_REDUCTION = False
    N_JOBS = -1

    df = load_data('Twitter_Data.csv', sample_size=SAMPLE_SIZE)
    df = clean_data(df)
    df = preprocess_text_data(df)
    df['sentiment'] = df['category'].map({1.0: 'Positive', 0.0: 'Neutral', -1.0: 'Negative'})

    generate_eda_plots(df)
    generate_wordclouds(df)

    X, y, tfidf = extract_features(df, use_dimensionality_reduction=DIM_REDUCTION)

    X, y = optimize_sample(df, X, y, sample_size=10000 if SAMPLE_SIZE is not None else len(df))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    
    print("Test set label distribution:")
    print(pd.Series(y_test).value_counts())

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    results, times = train_model(X_train, y_train, use_fast_svm=USE_FAST_SVM, n_jobs=N_JOBS)

    final_model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1',
                                     C=1, class_weight={-1: 2, 0: 1, 1: 1},
                                     random_state=42)
    final_model.fit(X_train, y_train)

    y_pred, y_proba, accuracy, pred_time = evaluate_model(final_model, X_test, y_test,
                                                          class_names=['Negative', 'Neutral', 'Positive'])

    total_time = time.time() - start_time
    print("\n===== Summary =====")
    print(f"Dataset size: {len(df)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Training times by model: {', '.join([f'{k}: {v:.2f}s' for k,v in times.items()])}")

if __name__ == '__main__':
    main()
