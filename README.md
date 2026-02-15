# Sports vs Politics News Classifier

**Course:** Natural Language Understanding (NLU)  
**Student:** Neer Modi  
**Roll No:** B23CS1043  
**Status:** ✓ Complete  

---

## Executive Summary

A comprehensive machine learning project that achieves **97.32% accuracy** classifying news articles as Sports or Politics using Natural Language Processing and text classification techniques.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** 🏆 | **97.32%** | **97.88%** | **99.09%** | **98.48%** |
| Naive Bayes | 96.64% | 96.67% | 99.59% | 98.11% |
| Logistic Regression | 96.58% | 96.51% | 99.71% | 98.08% |

---

## Project Overview

Complete NLP pipeline including:

✓ **Data Preparation** - Filtered 40,679 news articles (Sports & Politics)  
✓ **Text Preprocessing** - 6-step cleaning pipeline with NLTK  
✓ **Feature Engineering** - TF-IDF vectorization (5,000 features)  
✓ **Model Training** - Naive Bayes, Logistic Regression, Linear SVM  
✓ **Evaluation** - Comprehensive metrics and visualizations  
✓ **Documentation** - Complete report with analysis  

---

## Dataset

**Source:** News Category Dataset v3 (Kaggle)  
**Time Period:** 2012-2018  
**Publication:** HuffPost News

### Statistics
- Total articles analyzed: 40,679
- SPORTS articles: 5,077 (12.5%)
- POLITICS articles: 35,602 (87.5%)
- Training samples: 32,543
- Test samples: 8,136

---

## Folder Structure

```
sports-vs-politics-classifier/
├── data/
│   ├── raw/
│   │   └── News_Category_Dataset_v3.json        # Raw dataset
│   └── processed/
│       ├── processed_data.csv                    # Cleaned data
│       ├── X_train.pkl, X_test.pkl              # Features
│       ├── y_train.pkl, y_test.pkl              # Labels
│       └── tfidf_vectorizer.pkl                 # Vectorizer
├── src/
│   ├── prepare.py                               # Data preparation
│   ├── features.py                              # Feature engineering
│   ├── train.py                                 # Model training
│   └── evaluate.py                              # Evaluation
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   └── linear_svm.pkl
├── results/
│   ├── confusion_matrix_*.png
│   ├── metrics_heatmap.png
│   ├── metrics_comparison.png
│   └── metrics.csv
├── report/
│   └── report.md                                # Comprehensive report
├── docs/
│   ├── index.md                                 # GitHub Pages
│   ├── results.md
│   ├── models.md
│   └── methodology.md
├── requirements.txt
├── run_pipeline.py                              # Execute pipeline
├── _config.yml                                  # GitHub Pages config
└── README.md                                    # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download `News_Category_Dataset_v3.json` from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and place in `data/raw/`

### 3. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:
1. Load and filter the dataset
2. Preprocess text
3. Create TF-IDF features
4. Train all three models
5. Generate evaluation metrics
6. Create visualizations

---

## Preprocessing Pipeline

### 6-Step Text Cleaning

```python
1. Lowercase conversion        → "SPORTS" → "sports"
2. Special character removal   → "Sports!" → "Sports"
3. Tokenization                → "sports news" → ["sports", "news"]
4. Stopword removal            → remove common words (the, a, is)
5. Lemmatization               → "playing" → "play", "better" → "good"
6. Text reconstruction         → join tokens back together
```

### Example

```
BEFORE:
"Andrew McCutchen Wins 2012 NL MVP! Check https://example.com..."

AFTER:
"andrew mccutchen win nl mvp check"
```

---

## Feature Engineering

### TF-IDF Vectorization

```
Formula: TF-IDF(t,d) = TF(t,d) × IDF(t)

Configuration:
  - Max features: 5,000
  - Min document frequency: 2
  - Max document frequency: 95%
  - N-gram range: (1,1) - unigrams only

Output shape: (40,679 documents, 5,000 features)
```

### Top Features

**SPORTS Terms:**  
game, team, player, season, league, coach, win, score, championship, football

**POLITICS Terms:**  
president, congress, bill, senate, vote, trump, democrat, republican, amendment, legislation

---

## Machine Learning Models

### 1. Linear SVM (Recommended) 🏆

**Best Performance**
- Accuracy: 97.32%
- Precision: 97.88%
- Recall: 99.09%
- Training time: 0.044s

**Mathematical Formula:**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

**Why SVM?**
- Excellent for high-dimensional sparse data
- Maximum margin decision boundary
- Robust to outliers
- Fast inference

---

### 2. Naive Bayes

**Fast & Simple**
- Accuracy: 96.64%
- Training time: <0.01s (fastest)
- Good baseline model

**Mathematical Principle:**
$$P(\text{Class}|\text{Features}) = \frac{P(\text{Features}|\text{Class}) \times P(\text{Class})}{P(\text{Features})}$$

**Best for:** Real-time predictions, low-resource scenarios

---

### 3. Logistic Regression

**Balanced Performance**
- Accuracy: 96.58%
- Interpretable coefficients
- Good generalization

**Mathematical Formula:**
$$P(\text{Class}=1) = \frac{1}{1 + e^{-z}}$$

**Best for:** Explainability, feature importance analysis

---

## Results & Visualizations

### Confusion Matrix Analysis

**Linear SVM (Best Model):**
```
                  Predicted SPORTS  Predicted POLITICS
Actual SPORTS              862                153
Actual POLITICS             65               7056
```

**Interpretation:**
- True Positive Rate (Sports correctly identified): 85%
- True Negative Rate (Politics correctly identified): 99%
- Overall accuracy: 97.32%

### Generated Visualizations

1. **category_distribution.png** - Dataset breakdown (Sports vs Politics)
2. **feature_comparison.png** - Comparison of vectorization methods
3. **confusion_matrix_*.png** - Confusion matrices for all models
4. **metrics_heatmap.png** - Heatmap of all performance metrics
5. **metrics_comparison.png** - Bar chart comparing models

All visualizations saved to `results/` directory.

---

## Model Performance Comparison

### Accuracy Ranking
1. 🥇 Linear SVM: 97.32%
2. 🥈 Naive Bayes: 96.64%
3. 🥉 Logistic Regression: 96.58%

### Precision Ranking
1. 🥇 Linear SVM: 97.88%
2. 🥈 Naive Bayes: 96.67%
3. 🥉 Logistic Regression: 96.51%

### Speed Ranking
1. 🥇 Naive Bayes: <1ms
2. 🥈 Linear SVM: 44ms
3. 🥉 Logistic Regression: 69ms

---

## Using Trained Models

### Load and Predict

```python
import pickle
from pathlib import Path

# Load model and vectorizer
with open('models/linear_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Import preprocessing
from src.features import preprocess_text

# Make prediction
text = "Team wins championship game"
cleaned = preprocess_text(text)
X = vectorizer.transform([cleaned])
prediction = model.predict(X)[0]

# 0 = SPORTS, 1 = POLITICS
result = "SPORTS" if prediction == 0 else "POLITICS"
print(f"Predicted: {result}")
```

---

## Project Documentation

### Main Report
- **File:** `report/report.md`
- **Contains:** Complete analysis, methodology, results
- **Length:** Comprehensive academic report

### GitHub Pages
- **Location:** `docs/` folder
- **Pages:**
  - `index.md` - Homepage with project overview
  - `results.md` - Detailed results and metrics
  - `models.md` - Model descriptions and math
  - `methodology.md` - Technical methodology

To publish to GitHub Pages:
1. Create GitHub repository (public)
2. Push all files
3. Settings → Pages → Source: `/docs` folder
4. Site will be live in ~1-5 minutes

---

## Installation from Source

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git
cd sports-vs-politics-classifier
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset
- Place in: `data/raw/News_Category_Dataset_v3.json`

### 5. Run Pipeline
```bash
python run_pipeline.py
```

---

## Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | False positive rate |
| **Recall** | TP/(TP+FN) | False negative rate |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean (balanced) |

Where:
- **TP** = True Positives (correctly predicted)
- **TN** = True Negatives (correctly predicted)
- **FP** = False Positives (type I error)
- **FN** = False Negatives (type II error)

---

## Key Findings

✓ All models achieve >96% accuracy  
✓ Linear SVM performs best (97.32%)  
✓ Clear semantic separation between categories  
✓ TF-IDF features are highly discriminative  
✓ Minimal class imbalance effects  
✓ Models generalize well to test data  

---

## Future Improvements

1. **Advanced Models**
   - BERT/Transformer embeddings
   - Neural networks (LSTM, CNN)
   - Ensemble methods

2. **Optimization**
   - Hyperparameter tuning (GridSearch, Bayesian)
   - Cross-validation
   - Feature selection

3. **Deployment**
   - REST API (Flask/FastAPI)
   - Docker containerization
   - Model serving infrastructure

4. **Analysis**
   - Explainability (LIME, SHAP)
   - Error analysis
   - Domain adaptation

---

## References

- **Dataset:** [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Libraries:**
  - [Scikit-learn](https://scikit-learn.org/) - ML algorithms
  - [NLTK](https://www.nltk.org/) - NLP toolkit
  - [Pandas](https://pandas.pydata.org/) - Data manipulation
  - [NumPy](https://numpy.org/) - Numerical computing

---

## Project Information

**Status:** ✓ Complete  
**Last Updated:** February 15, 2026  
**Python Version:** 3.9+  
**License:** Educational Use  

**Student:** Neer Modi  
**Roll No:** B23CS1043  
**Course:** Natural Language Understanding (NLU)  

**Key Achievement:** 97.32% Accuracy with Linear SVM Classification

---

*For detailed analysis, see `report/report.md`*  
*For GitHub Pages publication, see `docs/` folder*  
*For code documentation, see inline comments in `src/` scripts*
