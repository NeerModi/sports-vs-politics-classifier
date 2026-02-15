# Sports vs Politics News Classification

**Course:** Natural Language Understanding (NLU)  
**Student:** Neer Modi  
**Roll No:** B23CS1043  
**Status:** ✓ Complete  

---

## 🎯 Project Overview

This project implements a **complete Natural Language Processing (NLP) pipeline** to automatically classify news articles into two categories: **Sports** and **Politics**. 

Using machine learning algorithms trained on **40,679 real-world news articles**, we achieved an impressive **97.32% accuracy** with Linear SVM.

---

## ⚡ Quick Stats

| Metric | Value |
|--------|-------|
| **Best Accuracy** | 97.32% |
| **Best Model** | Linear SVM |
| **Precision** | 97.88% |
| **Recall** | 99.09% |
| **F1-Score** | 98.48% |
| **Dataset Size** | 40,679 articles |
| **Training Samples** | 32,543 |
| **Test Samples** | 8,136 |

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** 🏆 | **97.32%** | **97.88%** | **99.09%** | **98.48%** |
| Naive Bayes | 96.64% | 96.67% | 99.59% | 98.11% |
| Logistic Regression | 96.58% | 96.51% | 99.71% | 98.08% |

---

## 📚 Documentation

Explore the project in detail:

- **[📈 Results & Metrics](results.md)** - Detailed performance analysis, confusion matrices, and visualizations
- **[🤖 Machine Learning Models](models.md)** - Mathematical foundations and technical descriptions
- **[🔬 Methodology](methodology.md)** - Complete pipeline explanation and preprocessing details

---

## 🎯 Key Features

✓ **Complete NLP Pipeline** - From raw JSON to trained models  
✓ **Multiple Models** - Comparison of 3 classification algorithms  
✓ **High Accuracy** - 97.32% on unseen test data  
✓ **Professional Documentation** - Comprehensive report with mathematical details  
✓ **Visualizations** - 6 charts and confusion matrices  
✓ **Production-Ready** - Trained models saved and ready to use  

---

## 📖 Dataset

**Source:** News Category Dataset v3 (Kaggle)  
**Time Period:** 2012-2018  
**Publication:** HuffPost News

### Class Distribution
- **SPORTS:** 5,077 articles (12.5%)
- **POLITICS:** 35,602 articles (87.5%)

---

## 🏗️ Project Architecture

```
Data Preparation → Feature Engineering → Model Training → Evaluation
     ↓                    ↓                     ↓              ↓
Load JSON          TF-IDF Features      Train 3 Models    Generate Metrics
Filter Categories   5,000 dimensions    Linear SVM        Confusion Matrices
Encode Labels       80:20 split          Naive Bayes      Visualizations
                                        Logistic Reg
```

---

## 🛠️ Technologies Used

- **Python 3.9+** - Programming language
- **Scikit-learn** - Machine learning algorithms
- **NLTK** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

---

## 📝 Preprocessing Pipeline

### Text Cleaning (6 Steps)

1. **Lowercase conversion** - Normalize text case
2. **Special character removal** - Keep only alphanumeric
3. **Tokenization** - Split into words
4. **Stopword removal** - Remove common words (the, a, is)
5. **Lemmatization** - Convert to base form (playing → play)
6. **Text reconstruction** - Rejoin processed tokens

### Example
```
BEFORE: "Andrew McCutchen Wins 2012 NL MVP! Check @MLB..."
AFTER:  "andrew mccutchen win nml mvp check"
```

---

## 🤖 Best Model: Linear SVM

### Performance Metrics
```
Test Accuracy:   97.32%
Precision:       97.88%
Recall:          99.09%
F1-Score:        98.48%
ROC-AUC:         98.73%
```

### Why Linear SVM?
- ✓ Excellent for high-dimensional sparse data
- ✓ Maximum margin decision boundary
- ✓ Fast inference time (0.0004s per sample)
- ✓ Robust generalization

### Confusion Matrix
```
                  Predicted SPORTS  Predicted POLITICS
Actual SPORTS              862                153
Actual POLITICS             65               7056
```

---

## 📁 Project Structure

```
sports-vs-politics-classifier/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed features
├── src/
│   ├── prepare.py              # Data preparation
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   └── evaluate.py             # Evaluation
├── models/
│   ├── linear_svm.pkl          # Best model
│   ├── naive_bayes.pkl
│   └── logistic_regression.pkl
├── results/
│   ├── confusion_matrix_*.png   # Visualizations
│   ├── metrics.csv
│   └── metrics_heatmap.png
├── report/
│   └── report.md               # Comprehensive report
├── docs/
│   ├── index.md               # This file
│   ├── results.md
│   ├── models.md
│   └── methodology.md
├── README.md                   # Quick start guide
├── requirements.txt
├── _config.yml                 # GitHub Pages config
└── run_pipeline.py             # Execute full pipeline
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git
cd sports-vs-politics-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline
```bash
python run_pipeline.py
```

This will:
1. Load and process the dataset
2. Extract features
3. Train all 3 models
4. Generate visualizations
5. Create performance reports

---

## 📊 Results Highlights

### Accuracy by Model
- **Linear SVM:** 97.32% (Best)
- **Naive Bayes:** 96.64%
- **Logistic Regression:** 96.58%

### Per-Class Performance (Linear SVM)
- **SPORTS:** 85% recall, 93% precision
- **POLITICS:** 99% recall, 98% precision

### Key Insights
- Clear vocabulary distinction between categories
- Minimal class imbalance effects
- All models generalize well to test data
- Linear SVM provides best overall performance

---

## 📖 Read More

For detailed information, see:

- **[Results & Metrics](results.md)** - Complete performance analysis
- **[Machine Learning Models](models.md)** - Model descriptions with mathematics
- **[Methodology](methodology.md)** - Technical pipeline details
- **[Full Report](../report/report.md)** - Comprehensive academic report

---

## 👤 Author

**Name:** Neer Modi  
**Roll No:** B23CS1043  
**Course:** Natural Language Understanding (NLU)  
**Date:** February 15, 2026  

---

## 📄 License

This project is open source and available for educational purposes.

---

**Project Status:** ✓ Complete  
**Website Built With:** Jekyll + GitHub Pages  
**Best Accuracy Achieved:** 97.32% (Linear SVM)

---
