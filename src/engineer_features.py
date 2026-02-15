"""
Feature Engineering Script
Preprocesses text, creates features using TF-IDF and BoW, performs train-test split
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess_batch


def load_data(data_dir):
    """Load processed data from CSV"""
    print("Loading processed data...")
    
    csv_path = data_dir / 'processed_data.csv'
    df = pd.read_csv(csv_path)
    
    print(f"Loaded: {csv_path}")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    return df


def preprocess_texts(texts):
    """Preprocess all texts"""
    print("\nPreprocessing texts...")
    
    processed_texts = preprocess_batch(texts)
    
    print(f"Processed {len(processed_texts)} texts")
    print(f"Sample preprocessed text: {processed_texts[0][:100]}...")
    
    return processed_texts


def create_tfidf_features(texts):
    """Create TF-IDF features"""
    print("\nCreating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )
    
    X_tfidf = vectorizer.fit_transform(texts)
    
    print(f"TF-IDF feature shape: {X_tfidf.shape}")
    print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")
    
    return X_tfidf, vectorizer


def create_bow_features(texts):
    """Create Bag of Words features"""
    print("\nCreating Bag of Words features...")
    
    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_bow = vectorizer.fit_transform(texts)
    
    print(f"BoW feature shape: {X_bow.shape}")
    
    return X_bow, vectorizer


def create_ngram_features(texts):
    """Create n-gram features"""
    print("\nCreating N-gram features...")
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=5,
        max_df=0.8,
        ngram_range=(2, 3),
        sublinear_tf=True,
        analyzer='word',
        stop_words='english'
    )
    
    X_ngrams = vectorizer.fit_transform(texts)
    
    print(f"N-gram feature shape: {X_ngrams.shape}")
    
    return X_ngrams, vectorizer


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets (stratified)"""
    print(f"\nSplitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  - Class {label}: {count} ({count/len(y_train)*100:.1f}%)")
    print(f"Test set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  - Class {label}: {count} ({count/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def save_features(X_train, X_test, y_train, y_test, vectorizer, output_dir, feature_name):
    """Save features to pickle files"""
    print(f"\nSaving {feature_name} features...")
    
    files = {
        'X_train.pkl': X_train,
        'X_test.pkl': X_test,
        'y_train.pkl': y_train,
        'y_test.pkl': y_test,
        f'{feature_name}_vectorizer.pkl': vectorizer
    }
    
    for filename, obj in files.items():
        filepath = output_dir / filename.replace('pkl', f'{feature_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {filepath}")
    
    # Also save generic versions (for compatibility)
    if feature_name == 'tfidf':
        generic_files = {
            'X_train.pkl': X_train,
            'X_test.pkl': X_test,
            'y_train.pkl': y_train,
            'y_test.pkl': y_test,
            'vectorizer.pkl': vectorizer
        }
        
        for filename, obj in generic_files.items():
            filepath = output_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            print(f"  Saved (generic): {filepath}")


def create_feature_comparison_viz(data_shapes, output_dir):
    """Create visualization comparing feature dimensions"""
    print("\nCreating feature comparison visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(data_shapes.keys())
    dimensions = list(data_shapes.values())
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(features, dimensions, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, dim in zip(bars, dimensions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(dim)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
    ax.set_title('Feature Dimension Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'feature_comparison.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution"""
    print("="*80)
    print("FEATURE ENGINEERING - SPORTS VS POLITICS CLASSIFIER")
    print("="*80)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    
    # Load data
    df = load_data(data_dir)
    
    texts = df['text'].values
    labels = df['label'].values
    
    # Preprocess texts
    processed_texts = preprocess_texts(texts)
    
    # Create features
    X_tfidf, tfidf_vec = create_tfidf_features(processed_texts)
    X_bow, bow_vec = create_bow_features(processed_texts)
    X_ngrams, ngrams_vec = create_ngram_features(processed_texts)
    
    # Store feature shapes for comparison
    feature_shapes = {
        'TF-IDF': X_tfidf.shape[1],
        'Bag of Words': X_bow.shape[1],
        'N-grams': X_ngrams.shape[1]
    }
    
    # Split data using TF-IDF (primary features)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, labels)
    
    # Save features
    save_features(X_train, X_test, y_train, y_test, tfidf_vec, data_dir, 'tfidf')
    save_features(X_train, X_test, y_train, y_test, bow_vec, data_dir, 'bow')
    
    # For n-grams, split separately since dimensions are different
    X_train_ng, X_test_ng, _, _ = split_data(X_ngrams, labels)
    save_features(X_train_ng, X_test_ng, y_train, y_test, ngrams_vec, data_dir, 'ngrams')
    
    # Create comparison visualization
    create_feature_comparison_viz(feature_shapes, data_dir)
    
    print("\n" + "="*80)
    print("✓ Feature Engineering Complete")
    print("="*80)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
