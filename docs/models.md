# Machine Learning Models

Detailed comparison and analysis of the three trained models for Sports vs Politics classification.

---

## 📊 Quick Comparison

| Aspect | Linear SVM | Naive Bayes | Logistic Regression |
|--------|-----------|------------|-------------------|
| **Accuracy** | **97.32%** ⭐ | 96.64% | 96.58% |
| **Training Speed** | 0.0499s | **0.0078s** ⭐ | 0.0849s |
| **Prediction Speed** | <1ms | <1ms | <1ms |
| **Interpretability** | Medium | High | High |
| **Complexity** | Medium | Low | Low |
| **Production Ready** | **Yes** ⭐ | Yes | Yes |

---

## 🏆 Model 1: Linear SVM (Recommended)

### Overview
Linear Support Vector Machine is a powerful classification algorithm that finds the optimal hyperplane to separate classes.

### Mathematical Foundation
Linear SVM solves:
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n} \xi_i$$

Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$

Where:
- $w$ = weight vector
- $b$ = bias term
- $C$ = regularization parameter
- $\xi_i$ = slack variables (allow some misclassification)

### Configuration
```python
LinearSVC(
    C=1.0,                    # Regularization strength
    max_iter=1000,           # Maximum iterations
    random_state=42,         # Reproducibility
    class_weight='balanced'   # Handle class imbalance
)
```

### Performance Metrics
```
Test Accuracy:    97.32%
Precision:        97.88%
Recall:           99.09%
F1-Score:         98.48%
ROC-AUC:          98.73%
Training Time:    0.0499 seconds
```

### Confusion Matrix
```
                    Predicted
                 SPORTS  POLITICS
Actual SPORTS       862      153
       POLITICS      65    7,056
```

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| SPORTS | 93% | 85% | 89% |
| POLITICS | 98% | 99% | 98% |

### Advantages
✓ **Highest accuracy** (97.32%)  
✓ **Best F1-score** (98.48%)  
✓ Excellent precision-recall balance  
✓ Good for imbalanced datasets  
✓ Memory efficient  
✓ Fast prediction (scales well)  

### Disadvantages
⚠ Less interpretable than linear models  
⚠ Requires feature scaling (TF-IDF handles this)  
⚠ Can be sensitive to hyperparameter tuning  

### When to Use
✅ Production deployment  
✅ When accuracy is critical  
✅ With imbalanced datasets  
✅ When you need fast, scalable solution  

### Use Cases
- **Real-time article classification**
- **Content recommendation systems**
- **Automated news categorization**
- **Feed filtering systems**

### Model File
**Location:** `models/linear_svm.pkl`  
**Size:** ~2.3 MB  
**Format:** Pickle (Python binary)

---

## 🤖 Model 2: Naive Bayes

### Overview
Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming feature independence.

### Mathematical Foundation
Bayes' Theorem:
$$P(Class|Features) = \frac{P(Features|Class) \cdot P(Class)}{P(Features)}$$

For Multinomial NB with text:
$$P(Class|Document) \propto P(Class) \prod_{i=1}^{d} P(feature_i|Class)$$

### Configuration
```python
MultinomialNB(
    alpha=1.0,              # Laplace smoothing
    fit_prior=True,        # Learn class priors
    class_prior=None       # Use empirical class frequencies
)
```

### Performance Metrics
```
Test Accuracy:    96.64%
Precision:        96.67%
Recall:           99.59%
F1-Score:         98.11%
ROC-AUC:          98.85%
Training Time:    0.0078 seconds (Fastest!)
```

### Confusion Matrix
```
                    Predicted
                 SPORTS  POLITICS
Actual SPORTS       771      244
       POLITICS      29    7,092
```

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| SPORTS | 96% | 76% | 85% |
| POLITICS | 97% | 100% | 98% |

### Advantages
✓ **Fastest training** (0.0078s)  
✓ **Highest recall** (99.59%)  
✓ Highly interpretable (can explain decisions)  
✓ Works well with sparse features  
✓ Probabilistic output (confidence scores)  
✓ Extremely lightweight  

### Disadvantages
⚠ Lower precision (96.67%) → More false positives  
⚠ Assumes feature independence (not fully true)  
⚠ Can underestimate probabilities  

### When to Use
✅ When recall is more important than precision  
✅ When model interpretability is needed  
✅ When fast training is critical  
✅ As a baseline/benchmark model  
✅ Resource-constrained environments  

### Use Cases
- **Baseline model for comparison**
- **Spam detection** (high recall preferred)
- **Quick prototyping**
- **Educational demonstrations**
- **Real-time systems with limited compute**

### Feature Importance
```
Top Features for SPORTS:
1. 'game' (strong indicator)
2. 'team' (strong indicator)
3. 'player' (strong indicator)
4. 'score' (strong indicator)
5. 'coach' (strong indicator)

Top Features for POLITICS:
1. 'trump' (strong indicator)
2. 'congress' (strong indicator)
3. 'law' (strong indicator)
4. 'senator' (strong indicator)
5. 'vote' (strong indicator)
```

### Model File
**Location:** `models/naive_bayes.pkl`  
**Size:** ~0.8 MB  
**Format:** Pickle (Python binary)

---

## 📊 Model 3: Logistic Regression

### Overview
Logistic Regression models the probability of a class using the logistic function.

### Mathematical Foundation
Logistic Function:
$$P(Class = 1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Loss Function (Binary Cross-Entropy):
$$J(w) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(h(x_i)) + (1-y_i)\log(1-h(x_i))]$$

### Configuration
```python
LogisticRegression(
    max_iter=1000,              # Maximum iterations
    random_state=42,            # Reproducibility
    class_weight='balanced',    # Handle class imbalance
    solver='lbfgs',            # Optimization algorithm
    multi_class='multinomial'  # For multi-class problems
)
```

### Performance Metrics
```
Test Accuracy:    96.58%
Precision:        96.51%
Recall:           99.71%
F1-Score:         98.08%
ROC-AUC:          98.81%
Training Time:    0.0849 seconds
```

### Confusion Matrix
```
                    Predicted
                 SPORTS  POLITICS
Actual SPORTS       758      257
       POLITICS      21    7,100
```

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| SPORTS | 97% | 75% | 85% |
| POLITICS | 97% | 100% | 98% |

### Advantages
✓ **Highest recall** (99.71%)  
✓ Highly interpretable (linear decision boundary)  
✓ Probabilistic predictions  
✓ Can extract feature coefficients  
✓ Well-calibrated probabilities  
✓ Fast prediction  

### Disadvantages
⚠ Lower accuracy than Linear SVM (96.58%)  
⚠ Assumes linear separability  
⚠ Not ideal for complex decision boundaries  

### When to Use
✅ When interpretability is critical  
✅ When you need probability estimates  
✅ When linear boundaries are appropriate  
✅ For exploratory analysis  
✅ When you need to explain decisions  

### Use Cases
- **Decision support systems** (need explainability)
- **Medical diagnosis** (need confidence scores)
- **Risk assessment** (probability critical)
- **Academic research** (interpretability needed)

### Feature Coefficients (Top)
```
Features Most Indicative of SPORTS:
1. 'game' (coefficient: +2.45)
2. 'team' (coefficient: +2.12)
3. 'player' (coefficient: +1.98)
4. 'score' (coefficient: +1.87)

Features Most Indicative of POLITICS:
1. 'trump' (coefficient: -3.21)
2. 'congress' (coefficient: -2.98)
3. 'law' (coefficient: -2.76)
4. 'senator' (coefficient: -2.54)
```

(Negative coefficients indicate POLITICS class)

### Model File
**Location:** `models/logistic_regression.pkl`  
**Size:** ~1.2 MB  
**Format:** Pickle (Python binary)

---

## 🔄 Model Training Details

### Data Used
- **Training samples:** 32,543
- **Test samples:** 8,136
- **Features:** 5,000 TF-IDF features
- **Classes:** 2 (SPORTS=0, POLITICS=1)

### Hyperparameters Used

**All models trained with:**
- Random state: 42 (reproducibility)
- Class weight: balanced (handle imbalance)
- Max iterations: 1000

**Linear SVM specific:**
```python
C=1.0  # Regularization strength
class_weight='balanced'
max_iter=1000
```

**Naive Bayes specific:**
```python
alpha=1.0  # Laplace smoothing
fit_prior=True
```

**Logistic Regression specific:**
```python
max_iter=1000
solver='lbfgs'
class_weight='balanced'
```

---

## 📈 Performance Comparison

### Accuracy by Model
```
Linear SVM:       97.32% ⭐
Naive Bayes:      96.64%
Logistic Regr:    96.58%
```

### Precision by Model
```
Linear SVM:       97.88% ⭐
Logistic Regr:    96.51%
Naive Bayes:      96.67%
```

### Recall by Model
```
Logistic Regr:    99.71% ⭐
Naive Bayes:      99.59%
Linear SVM:       99.09%
```

### F1-Score by Model
```
Linear SVM:       98.48% ⭐
Naive Bayes:      98.11%
Logistic Regr:    98.08%
```

---

## 🎯 Model Selection Guide

### Choose Linear SVM if:
- ✓ Maximum accuracy is needed
- ✓ Production deployment
- ✓ Balanced precision & recall needed
- ✓ Imbalanced dataset
- ✓ Speed and accuracy both important

### Choose Naive Bayes if:
- ✓ Speed is critical (0.0078s training)
- ✓ Model simplicity desired
- ✓ Interpretability very important
- ✓ Limited computational resources
- ✓ Need baseline comparison

### Choose Logistic Regression if:
- ✓ Interpretability is crucial
- ✓ Need probability estimates
- ✓ Linear boundaries appropriate
- ✓ Feature importance analysis needed
- ✓ Regulatory compliance requires explanation

---

## 🔮 Improvement Opportunities

### Model Ensembling
Combine all three models:
```python
# Voting Classifier
ensemble = VotingClassifier(
    estimators=[('svm', svm_model), 
                ('nb', nb_model), 
                ('lr', lr_model)],
    voting='soft'  # Use probability averaging
)
# Expected improvement: +0.5-1% accuracy
```

### Hyperparameter Tuning
```python
# Grid Search Example
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
# Expected improvement: +0.5-1.5% accuracy
```

### Feature Engineering
- Add word embeddings (Word2Vec, GloVe)
- Include semantic features
- Try different n-gram combinations
- Expected improvement: +1-2% accuracy

### Data Augmentation
- Collect more training data
- Use data synthesis techniques
- Expected improvement: +2-3% accuracy

---

## 📝 Using Models for Predictions

### Load and Predict with Linear SVM

```python
import pickle
from src.preprocess import clean_text

# Load model and vectorizer
model = pickle.load(open('models/linear_svm.pkl', 'rb'))
vectorizer = pickle.load(open('data/processed/vectorizer.pkl', 'rb'))

# New article
article = "Team wins championship game with amazing performance"

# Preprocess
cleaned = clean_text(article)

# Vectorize
X = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(X)[0]
probability = model.decision_function(X)[0]

# Output
category = 'SPORTS' if prediction == 0 else 'POLITICS'
print(f"Category: {category}")
print(f"Confidence: {abs(probability):.2f}")
```

### Get Probability Estimates

```python
# For Naive Bayes (has probabilities)
nb_model = pickle.load(open('models/naive_bayes.pkl', 'rb'))
probabilities = nb_model.predict_proba(X)[0]
print(f"P(SPORTS): {probabilities[0]:.4f}")
print(f"P(POLITICS): {probabilities[1]:.4f}")
```

---

## 📊 Model Files

| Model | File | Size | Format | Status |
|-------|------|------|--------|--------|
| Linear SVM | `models/linear_svm.pkl` | 2.3 MB | Pickle | ✓ Ready |
| Naive Bayes | `models/naive_bayes.pkl` | 0.8 MB | Pickle | ✓ Ready |
| Logistic Regression | `models/logistic_regression.pkl` | 1.2 MB | Pickle | ✓ Ready |

---

## 🔗 Related Pages

- **[Results](results.md)** - Detailed performance metrics
- **[Methodology](methodology.md)** - How models were trained
- **[Home](index.md)** - Project overview

---

**Last Updated:** February 15, 2026  
**Status:** ✓ All Models Production Ready
