import sys
import os
import pickle
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_data():
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    with open(data_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes classifier."""
    print("TRAINING: Naive Bayes (Multinomial)")
    print("=" * 80)
    
    model = MultinomialNB()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training time: {training_time:.4f} seconds")
    
    return model, training_time


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier."""
    print("TRAINING: Logistic Regression")
    print("=" * 80)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training time: {training_time:.4f} seconds")
    
    return model, training_time


def train_linear_svm(X_train, y_train):
    """Train Linear SVM classifier."""
    print("\n" + "=" * 80)
    print("=" * 80)
    
    model = LinearSVC(max_iter=2000, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training time: {training_time:.4f} seconds")
    
    return model, training_time


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model on training and test data."""
    print(f"\n{model_name} Evaluation:")
    print("-" * 80)
    
    # Training predictions
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    train_pred_time = time.time() - start_time
    
    # Test predictions
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_pred_time = time.time() - start_time
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Prediction Time: {train_pred_time:.4f} seconds")
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print(f"\nTest Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test Prediction Time: {test_pred_time:.4f} seconds")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['SPORTS', 'POLITICS']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'train_time': train_pred_time,
        'test_time': test_pred_time,
        'y_test_pred': y_test_pred,
        'y_true': y_test,
        'cm': cm
    }


def save_models(models, data_dir):
    """Save trained models."""
    models_dir = data_dir.parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for model, name in models:
        model_path = models_dir / f'{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {name} to {model_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("SPORTS VS POLITICS CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Train models
    nb_model, nb_train_time = train_naive_bayes(X_train, y_train)
    lr_model, lr_train_time = train_logistic_regression(X_train, y_train)
    svm_model, svm_train_time = train_linear_svm(X_train, y_train)
    
    # Evaluate models
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = []
    results.append(evaluate_model(nb_model, X_train, X_test, y_train, y_test, "Naive Bayes"))
    results.append(evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Logistic Regression"))
    results.append(evaluate_model(svm_model, X_train, X_test, y_train, y_test, "Linear SVM"))
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    save_models([(nb_model, 'naive_bayes'), 
                 (lr_model, 'logistic_regression'), 
                 (svm_model, 'linear_svm')], 
                data_dir)
    
    # Save results summary
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Train Accuracy': r['train_accuracy'],
        'Test Accuracy': r['test_accuracy'],
        'Precision': r['test_precision'],
        'Recall': r['test_recall'],
        'F1-Score': r['test_f1'],
        'Training Time (s)': [nb_train_time, lr_train_time, svm_train_time][results.index(r)],
    } for i, r in enumerate(results)])
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
