# Twitter Sentiment Analysis

## Project Description
This project performs sentiment analysis on Twitter data to classify tweets into Positive, Neutral, and Negative sentiment categories. The implementation includes comprehensive data preprocessing, exploratory analysis, feature engineering, and machine learning model training to understand public sentiment patterns, particularly in political discourse.

## 1. Project Objective
Analyze and classify Twitter sentiment to understand public opinion patterns, with particular focus on political discourse. The goal is to build an accurate sentiment classification system that can identify key themes and topics associated with different sentiment polarities.

## 2. Dataset Information
- **Source**: Twitter data (`Twitter_Data.csv`)
- **Records**: 162,980 tweets (after cleaning: 162,969)
- **Features**:
  - `clean_text`: Preprocessed tweet text
  - `category`: Sentiment label (-1.0: Negative, 0.0: Neutral, 1.0: Positive)
- **Sentiment Distribution**:
  - Positive: 44.3% (72,249 tweets)
  - Neutral: 33.9% (55,211 tweets) 
  - Negative: 21.8% (35,509 tweets)

## 3. Methodology
- **Preprocessing**: Text cleaning, lowercasing, URL removal, special character removal, tokenization, lemmatization, and stopword removal
- **Feature Extraction**: TF-IDF vectorization with 5,000 features and n-gram range (1,2)
- **Model Training**: Compared multiple algorithms including Logistic Regression, Naive Bayes, Random Forest, and SVM
- **Model Selection**: Logistic Regression with L1 regularization and class weighting
- **Evaluation**: 3-fold cross-validation, stratified train-test split (80-20)

## 4. Model Performance

### Comparative Model Results:
| Model | Mean F1 Score | Training Time |
|-------|---------------|---------------|
| Logistic Regression | 0.8616 (±0.0032) | 4.74s |
| Naive Bayes | 0.7132 (±0.0034) | 1.40s |
| Random Forest | 0.8339 (±0.0035) | 43.26s |
| SVM (SGD) | 0.8268 (±0.0030) | 1.83s |

### Final Model Performance (Test Set: 32,594 samples):
- **Accuracy**: 88.40%
- **Prediction Time**: 0.0186 seconds

#### Detailed Classification Report:
| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative | 0.84 | 0.80 | 0.82 | 7,102 |
| Neutral | 0.85 | 0.96 | 0.90 | 11,042 |
| Positive | 0.94 | 0.87 | 0.90 | 14,450 |

## 5. Key Insights from Sentiment Analysis

### Positive Sentiment Themes
- **Dominant words**: `good`, `better`, `need`, `great`, `time`, `well`, `nation`, `leader`
- **Key figures**: `modi` (Narendra Modi)
- **Themes**: National progress, leadership appreciation, government initiatives

### Neutral Sentiment Themes  
- **Dominant words**: `people`, `work`, `govern`, `vote`, `time`, `today`, `bjp`, `congress`
- **Key phrases**: `surgical strike` (factual military reference)
- **Themes**: Factual reporting, procedural discussions, neutral political commentary

### Negative Sentiment Themes
- **Dominant words**: `wrong`, `dont`, `black money`, `want`
- **Key figures**: `modi` (primary subject of criticism), `congress`, `pakistan`
- **Themes**: Criticism of policies, governance issues, political opposition

### Behavioral Patterns
- **Political focus**: Significant discussion centered around political figures and parties
- **Emotional intensity**: Negative tweets show strong emotional language and criticism
- **Issue-based discourse**: Positive tweets focus on achievements, negative on problems
- **Neutral reporting**: Fact-based tweets without strong emotional alignment

## 6. Algorithm Implementation

### Feature Engineering
- TF-IDF Vectorization with 5,000 features
- N-gram range: (1,2) to capture phrases and context
- Text length feature added for additional signal

### Model Selection
- **Final algorithm**: Logistic Regression with L1 regularization
- **Hyperparameters**: solver='liblinear', penalty='l1', C=1
- **Class weights**: {-1: 2, 0: 1, 1: 1} to handle class imbalance
- **Training time**: 4.74 seconds for full dataset

### Evaluation Metrics
- Primary metric: Weighted F1-score
- Additional metrics: Precision, Recall, Accuracy
- Cross-validation: 3-fold stratified validation
- Visual evaluation: ROC curves, Precision-Recall curves, Confusion Matrix

## 7. Business and Research Implications

### Political Analysis Applications
- **Public opinion tracking**: Real-time sentiment monitoring for political figures
- **Policy feedback**: Understanding public reaction to government initiatives
- **Election analysis**: Sentiment trends during election periods
- **Crisis management**: Identifying emerging negative sentiment patterns

### Media and Journalism
- **Trend identification**: Discovering emerging topics and public concerns
- **Bias detection**: Analyzing sentiment patterns across different media outlets
- **Audience engagement**: Understanding what content resonates with different segments

### Commercial Applications
- **Brand monitoring**: Adaptable to corporate reputation management
- **Market research**: Understanding consumer sentiment about products/services
- **Customer service**: Identifying and addressing negative feedback quickly

## 8. Technical Implementation Insights

### Data Challenges
- **Class imbalance**: Addressed through class weighting in Logistic Regression
- **Text variability**: Handled through comprehensive preprocessing pipeline
- **Feature dimensionality**: Managed through TF-IDF feature selection

### Performance Optimization
- **Efficient algorithms**: Selected models based on performance-speed tradeoff
- **Parallel processing**: Utilized multi-core processing where available
- **Memory management**: Implemented sampling option for large datasets

### Visualization Approach
- **Word clouds**: For intuitive understanding of sentiment themes
- **Performance metrics**: Comparative charts for model evaluation
- **Statistical distributions**: For data understanding and quality assessment

## 9. Future Work
- Implement deep learning approaches (LSTMs, Transformers) for improved context understanding
- Add real-time Twitter API integration for live sentiment analysis
- Develop multi-lingual sentiment analysis capabilities
- Incorporate topic modeling to identify specific discussion themes within sentiments
- Create dashboard interface for interactive exploration of results
- Implement temporal analysis to track sentiment trends over time

## 10. Visualizations

The project generates comprehensive visualizations including:
- Sentiment distribution charts (bar and pie)
- Text length distribution by sentiment
- Word clouds for each sentiment category
- Model performance comparison charts
- Training time comparison visualizations
- Confusion matrix
- ROC curves and Precision-Recall curves

For complete details, analysis, and interpretation of all visualizations, please see the dedicated visualization document:

**[Comprehensive Visualizations Analysis](VISUALIZATIONS.md)**
---

## Installation Instructions
This project requires Python 3.x and the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

Install dependencies via pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

Download required NLTK datasets:
```bash
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## How to Run This Project
1. Clone or download this repository.
2. Place your dataset CSV file (Twitter_Data.csv) in the project directory directory or specify the file path in the code.
3. Run the main script:

bash
python sentiment_analysis.py
4. The script will:
    - Load and preprocess the data  
    - Generate exploratory visualizations 
    - Create word clouds for each sentiment  
    - Extract features using TF-IDF  
    - Train and compare multiple models
    - Evaluate the best model on test data
    - Generate performance visualizations
5. View outputs in the console and saved plots

---

## Contact
For questions or collaboration, please reach out to me:

- **Name:** Ghanashyam T V  
- **Email:** [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)  
- **LinkedIn:** [www.linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring this project! Feel free to open issues or pull requests for improvements.

