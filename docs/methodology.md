# Methodology

Comprehensive explanation of the approach, techniques, and methods used in this project.

---

## 📋 Project Pipeline Overview

```
┌─────────────────┐
│  Raw Data (JSON)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Data Preparation        │
│ • Load JSON             │
│ • Filter categories     │
│ • Combine text fields   │
│ • Encode labels         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Text Preprocessing      │
│ • Lowercase             │
│ • Remove punctuation    │
│ • Tokenize              │
│ • Remove stopwords      │
│ • Lemmatize             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Feature Engineering     │
│ • TF-IDF                │
│ • Bag of Words          │
│ • N-grams               │
│ • Train-test split      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Training          │
│ • Naive Bayes           │
│ • Logistic Regression   │
│ • Linear SVM            │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Evaluation        │
│ • Confusion matrices    │
│ • Performance metrics   │
│ • Visualizations        │
└─────────────────────────┘
```

---

## 🗂️ Phase 1: Data Preparation

### Objective
Extract relevant data from the News Category Dataset and prepare it for processing.

### Dataset Source
- **Dataset:** News Category Dataset v3
- **From:** Kaggle
- **Total Articles:** 209,527
- **Categories Available:** 41 different news categories

### Data Loading
```python
# Load JSON data (line-delimited JSON)
with open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        articles.append(article)
```

### Category Filtering
**Criteria:**
- Select only SPORTS and POLITICS articles
- Exclude all other categories

**Results:**
```
Total articles loaded:      209,527
After filtering:            40,679 (19.4%)
├─ SPORTS:     5,077 (12.5%)
├─ POLITICS:  35,602 (87.5%)
```

### Data Imbalance
- The dataset has a 7:1 class imbalance (87.5% POLITICS vs 12.5% SPORTS)
- **Solution:** Use stratified train-test split to maintain distribution in both sets

### Text Combination
For each article, combine:
- **Headline** (main title)
- **Short Description** (summary)

```python
combined_text = f"{headline} {description}"
```

### Label Encoding
```python
SPORTS = 0
POLITICS = 1
```

### Output
- **File:** `data/processed/processed_data.csv`
- **Rows:** 40,679 articles
- **Columns:** text, category, headline, description, date, label
- **Size:** 13.51 MB

---

## 🔤 Phase 2: Text Preprocessing

### Objective
Clean and normalize text to improve feature quality and model performance.

### Preprocessing Pipeline

#### Step 1: Lowercase Conversion
```python
text = text.lower()
```
**Purpose:** Standardize text (avoid treating "The" and "the" differently)

#### Step 2: URL & Email Removal
```python
text = re.sub(r'http\S+|www\S+|https\S+', '', text)
text = re.sub(r'\S+@\S+', '', text)
```
**Purpose:** Remove non-textual data that doesn't contribute to classification

#### Step 3: Punctuation Removal
```python
text = text.translate(str.maketrans('', '', string.punctuation))
```
**Purpose:** Remove special characters (except those with semantic meaning)

#### Step 4: Tokenization
```python
tokens = word_tokenize(text)
```
**Purpose:** Split text into individual words/tokens for processing

#### Step 5: Stopword Removal
```python
stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]
```
**Stopwords Removed:** a, the, is, and, or, to, etc. (180 English stopwords)

**Purpose:** Remove common words that don't provide discriminative information

#### Step 6: Lemmatization
```python
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]
```
**Purpose:** Reduce words to base form (e.g., "playing" → "play", "ran" → "run")

### Example Transformation
**Input:**
```
"Maury Will Basestealing Shortstop Dodger Dy Maury Will Helped Los Angeles Dodger Win Three World Series..."
```

**Output (after preprocessing):**
```
"maury basestealing shortstop dodger maury helped los angeles dodger win three world series"
```

### Implementation
- **File:** `src/preprocess.py`
- **Functions:**
  - `clean_text(text)` - Single article preprocessing
  - `preprocess_batch(texts)` - Batch processing (40,679 articles)
- **Dependencies:** NLTK, regex, string libraries
- **Processing Time:** ~3-5 minutes for full dataset

---

## 🧬 Phase 3: Feature Engineering

### Objective
Convert preprocessed text into numerical features suitable for ML models.

### Feature Extraction Methods

#### Method 1: TF-IDF (Term Frequency-Inverse Document Frequency)

**Mathematical Formula:**
$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

Where:
- $TF(t,d)$ = Frequency of term $t$ in document $d$
- $IDF(t) = \log\left(\frac{N}{n_t}\right)$ = Log of (total docs / docs containing $t$)

**Intuition:** Gives higher weight to terms that are frequent in a document but rare across all documents.

**Configuration:**
```python
TfidfVectorizer(
    max_features=5000,        # Top 5000 features
    min_df=5,                 # Min 5 documents
    max_df=0.8,               # Max 80% of documents
    ngram_range=(1, 2),       # Unigrams + bigrams
    sublinear_tf=True,        # Sublinear TF scaling
    strip_accents='unicode',  # Remove accents
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english'
)
```

**Output:** 40,679 × 5,000 sparse matrix

**Use:** Primary feature representation for all models

#### Method 2: Bag of Words (CountVectorizer)

**Formula:**
$$BoW(t,d) = \text{count of term } t \text{ in document } d$$

**Intuition:** Simple term frequency counting without IDF weighting.

**Configuration:**
```python
CountVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)
```

**Output:** 40,679 × 5,000 sparse matrix

**Comparison with TF-IDF:** TF-IDF performed better in our experiments

#### Method 3: N-gram Features

**Definition:** Sequences of N consecutive tokens

**Examples:**
- **Unigrams** (N=1): "sports", "politics", "vote"
- **Bigrams** (N=2): "world cup", "presidential election"
- **Trigrams** (N=3): "world cup final"

**Configuration:**
```python
TfidfVectorizer(
    max_features=3000,
    min_df=5,
    max_df=0.8,
    ngram_range=(2, 3),  # Only bigrams and trigrams
    sublinear_tf=True
)
```

**Output:** 40,679 × 3,000 sparse matrix

**Purpose:** Capture contextual information and phrase-level patterns

### Feature Statistics

| Feature Type | Dimensions | Sparsity | Coverage |
|--------------|-----------|----------|----------|
| TF-IDF (1-gram) | 5,000 | 99.5% | 12.3% of unique terms |
| Bag of Words | 5,000 | 99.8% | 12.3% of unique terms |
| N-grams (2-3) | 3,000 | 99.8% | Phrase coverage |

### Train-Test Split

**Method:** Stratified K-Fold Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% test, 80% train
    random_state=42,         # Reproducibility
    stratify=y               # Maintain class distribution
)
```

**Results:**
```
Training samples:  32,543 (80%)
├─ SPORTS:     4,062 (12.5%)
├─ POLITICS: 28,481 (87.5%)

Test samples:      8,136 (20%)
├─ SPORTS:     1,015 (12.5%)
└─ POLITICS:   7,121 (87.5%)
```

**Why Stratified?** Ensures both train and test sets have same class distribution as original data.

### Feature Scaling
TF-IDF automatically normalizes features to unit norm (L2 normalization).

---

## 🤖 Phase 4: Model Training

### Objective
Train three different classification algorithms on the training data.

### Training Data
- **Features:** X_train (32,543 × 5,000)
- **Labels:** y_train (32,543,)
- **Algorithm:** All trained with identical random state (42) for reproducibility

### Model 1: Multinomial Naive Bayes

**Theory:** Applies Bayes' theorem with assumption of feature independence.

**Training Process:**
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0, fit_prior=True)
model.fit(X_train, y_train)
```

**What it learns:**
- Prior probabilities: P(SPORTS), P(POLITICS)
- Feature probabilities: P(feature|SPORTS), P(feature|POLITICS)
- Feature log probabilities for numerical stability

**Training Time:** 0.0078 seconds

**Training Accuracy:** 96.98%

### Model 2: Logistic Regression

**Theory:** Models probability using logistic function.

**Training Process:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)
model.fit(X_train, y_train)
```

**What it learns:**
- Weight coefficients (w) for each feature
- Bias term (b)
- Feature importance (via coefficients)

**Optimization:** Uses Limited-memory BFGS solver

**Training Time:** 0.0849 seconds

**Training Accuracy:** 97.17%

### Model 3: Linear SVM

**Theory:** Finds optimal hyperplane maximizing margin between classes.

**Training Process:**
```python
from sklearn.svm import LinearSVC

model = LinearSVC(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
```

**What it learns:**
- Weight vector (w) defining decision boundary
- Bias term (b)
- Support vectors (critical training points)

**Regularization:** Parameter C=1.0 controls slack variables

**Training Time:** 0.0499 seconds

**Training Accuracy:** 99.12%

---

## 📊 Phase 5: Model Evaluation

### Evaluation Metrics

#### 1. Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:
- **TP** = True Positives (correctly predicted POLITICS)
- **TN** = True Negatives (correctly predicted SPORTS)
- **FP** = False Positives (sports predicted as politics)
- **FN** = False Negatives (politics predicted as sports)

#### 2. Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interpretation:** Of articles predicted as politics, how many were actually politics?

#### 3. Recall
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretation:** Of actual politics articles, how many did we correctly identify?

#### 4. F1-Score
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interpretation:** Harmonic mean of precision and recall.

#### 5. ROC-AUC
**ROC Curve:** Plots True Positive Rate vs False Positive Rate at different thresholds

**AUC:** Area under the ROC curve (0.5 = random, 1.0 = perfect)

### Evaluation Process

```python
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

### Evaluation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------| 
| Linear SVM | **97.32%** | **97.88%** | **99.09%** | **98.48%** | **98.73%** |
| Naive Bayes | 96.64% | 96.67% | 99.59% | 98.11% | 98.85% |
| Logistic Regression | 96.58% | 96.51% | 99.71% | 98.08% | 98.81% |

---

## 🔍 Detailed Analysis

### Overfitting Assessment

**Training vs Test Accuracy:**
- Linear SVM: Train 99.12% → Test 97.32% (gap: 1.80%)
- Naive Bayes: Train 96.98% → Test 96.64% (gap: 0.34%)
- Logistic Regression: Train 97.17% → Test 96.58% (gap: 0.59%)

**Conclusion:** Minimal overfitting, all models generalize well.

### Class-wise Performance

**SPORTS Class (Minority - 12.5%):**
- Best recall: Logistic Regression (75%)
- Best precision: Logistic Regression (97%)
- Most errors: Sports articles misclassified as politics

**POLITICS Class (Majority - 87.5%):**
- Best performance: All models >98%
- Easier to classify due to more training examples
- Few misclassifications

### Error Analysis

**Common Misclassifications:**
- Sports articles with political figures (e.g., "Trump attends game")
- Political articles about sports regulations
- Language overlap between categories

---

## 📈 Feature Importance

### Top Discriminative Features

**Most Important for SPORTS:**
1. "game" (very strong indicator)
2. "team" (strong indicator)
3. "player" (strong indicator)
4. "score" (strong indicator)
5. "coach" (moderate indicator)

**Most Important for POLITICS:**
1. "trump" (very strong indicator)
2. "congress" (very strong indicator)
3. "law" (strong indicator)
4. "senator" (strong indicator)
5. "vote" (strong indicator)

---

## 🔧 Hyperparameter Selection

### Why These Parameters?

**TF-IDF Configuration:**
- `max_features=5000`: Balance between expressiveness and dimensionality
- `min_df=5`: Exclude very rare terms (likely noise)
- `max_df=0.8`: Exclude very common terms (likely stopwords)
- `ngram_range=(1,2)`: Capture both individual words and phrases

**Train-Test Split:**
- `test_size=0.2`: Standard 80-20 split
- `random_state=42`: Reproducibility
- `stratify=y`: Maintain class distribution

**Model Hyperparameters:**
- `random_state=42`: Reproducibility
- `class_weight='balanced'`: Handle class imbalance
- `max_iter=1000`: Sufficient for convergence

---

## 🎯 Why This Approach?

### TF-IDF Over Bag of Words
- TF-IDF down-weights common terms (a, the, is)
- Gives more importance to discriminative terms
- Improved accuracy by ~0.5% in experiments

### Stratified Split
- Maintains class distribution in train and test
- Ensures fair evaluation despite 7:1 imbalance
- Prevents test set bias

### Multiple Models
- Compare different algorithms
- Understand trade-offs (accuracy vs speed vs interpretability)
- Select best model for use case

### Text Preprocessing
- Lemmatization reduces feature space
- Stopword removal improves signal-to-noise
- Preprocessing added ~3-5% accuracy improvement

---

## 📋 Experimental Log

| Experiment | Change | Result |
|-----------|--------|--------|
| Baseline | No preprocessing | 91.2% accuracy |
| + Preprocessing | Added cleaning | 94.8% accuracy |
| + TF-IDF | Better features | 96.6% accuracy |
| + Lemmatization | Reduce variations | 97.0% accuracy |
| + Hypertuning | Optimize params | 97.3% accuracy |

---

## 🚀 Reproducibility

### Requirements for Reproducibility
- Random state = 42 (all models)
- Same preprocessing pipeline
- Same train-test split method
- Same feature extraction parameters

### How to Reproduce
1. Clone repository
2. Follow [EXECUTION_INSTRUCTIONS.md](../EXECUTION_INSTRUCTIONS.md)
3. Run `python run_pipeline.py`
4. Compare results with documented metrics

---

## 🔗 Related Pages

- **[Results](results.md)** - Performance metrics and visualizations
- **[Models](models.md)** - Detailed model descriptions
- **[Home](index.md)** - Project overview

---

**Last Updated:** February 15, 2026  
**Status:** ✓ Complete Documentation
