"""
Model Evaluation and Comparison Script

Evaluates trained models and generates:
- Confusion matrices
- Performance metrics comparison
- Visualizations
- Results summary CSV
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data_and_models():
    """Load test data and trained models."""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Load data
    with open(data_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Load models
    models = {}
    model_names = ['naive_bayes', 'logistic_regression', 'linear_svm']
    for name in model_names:
        with open(models_dir / f'{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    return X_test, y_test, models


def generate_confusion_matrices(X_test, y_test, models):
    """Generate and save confusion matrices for each model."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    confusion_matrices = {}
    
    for model_name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm
        
        # Create and save visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['SPORTS', 'POLITICS'],
                   yticklabels=['SPORTS', 'POLITICS'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        fig_name = f'confusion_matrix_{model_name}.png'
        plt.savefig(results_dir / fig_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {fig_name}")
    
    return confusion_matrices


def calculate_metrics(X_test, y_test, models):
    """Calculate evaluation metrics for all models."""
    metrics_list = []
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Try to calculate ROC-AUC (may not work for all models)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = None
            
            if y_proba is not None:
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        metrics_list.append({
            'Model': model_name.replace('_', ' ').title(),
            'Features': 'TF-IDF (unigrams)',
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4) if roc_auc else 'N/A'
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def save_metrics_csv(metrics_df):
    """Save metrics comparison to CSV."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / 'metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Metrics saved to {csv_path}")
    print("\nMetrics Comparison Table:")
    print(metrics_df.to_string(index=False))


def create_metrics_comparison_viz(metrics_df):
    """Create visualization comparing metrics across models."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Select numeric columns
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_numeric = metrics_df[['Model'] + numeric_cols].copy()
    
    # Create heatmap
    plt.figure(figsize=(10, 5))
    data_for_heatmap = metrics_numeric.set_index('Model')[numeric_cols].values
    sns.heatmap(data_for_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
               xticklabels=numeric_cols,
               yticklabels=metrics_numeric['Model'],
               cbar_kws={'label': 'Score'}, vmin=0.7, vmax=1.0)
    plt.title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Model')
    plt.xlabel('Metric')
    plt.tight_layout()
    
    fig_path = results_dir / 'metrics_heatmap.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics heatmap saved to {fig_path}")
    
    # Create bar chart comparing accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        metric_values = metrics_df[metric].values
        bars = ax.bar(metrics_df['Model'], metric_values, color=colors)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim([0.7, 1.0])
        ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = results_dir / 'metrics_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison chart saved to {fig_path}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 80)
    print("SPORTS VS POLITICS CLASSIFICATION - MODEL EVALUATION")
    print("=" * 80)
    
    # Load data and models
    print("\nLoading data and models...")
    X_test, y_test, models = load_data_and_models()
    print(f"✓ Test set loaded: {X_test.shape}")
    print(f"✓ Models loaded: {list(models.keys())}")
    
    # Generate confusion matrices
    print("\nGenerating confusion matrices...")
    confusion_matrices = generate_confusion_matrices(X_test, y_test, models)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_df = calculate_metrics(X_test, y_test, models)
    
    # Save metrics
    print("\nSaving results...")
    save_metrics_csv(metrics_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_metrics_comparison_viz(metrics_df)
    
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/confusion_matrix_*.png")
    print("  - results/metrics.csv")
    print("  - results/metrics_heatmap.png")
    print("  - results/metrics_comparison.png")


if __name__ == "__main__":
    main()
