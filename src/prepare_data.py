import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def load_and_filter_data(json_path):
    print("Loading dataset from JSON...")
    
    articles = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            articles.append(article)
    
    print(f"Total articles loaded: {len(articles)}")
    
    # Filter for SPORTS and POLITICS
    sports_politics = [
        article for article in articles 
        if article.get('category') in ['SPORTS', 'POLITICS']
    ]
    
    print(f"After filtering (SPORTS + POLITICS): {len(sports_politics)}")
    print(f"  - SPORTS: {sum(1 for a in sports_politics if a['category'] == 'SPORTS')}")
    print(f"  - POLITICS: {sum(1 for a in sports_politics if a['category'] == 'POLITICS')}")
    
    return sports_politics


def create_dataframe(articles):
    print("\nCreating DataFrame...")
    
    data = []
    for article in articles:
        # Combine headline and description
        description = article.get('short_description', '')
        combined_text = f"{headline} {description}".strip()
        
        data.append({
            'text': combined_text,
            'category': article.get('category'),
            'headline': headline,
            'description': description,
            'date': article.get('date', '')
        })
    
    df = pd.DataFrame(data)
    return df


def encode_labels(df):
    """Encode categorical labels to numeric"""
    print("Encoding labels...")
    TS': 0, 'POLITICS': 1}
    df['label'] = df['category'].map(label_mapping)
    
    print(f"Label mapping: {label_mapping}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    return df, label_mapping


def create_visualizations(df, output_dir):
    """Create category distribution visualizations"""
    print("\nCreating visualizations...")
    print("\nCreating visualizations...")
    
    # Count plot
    category_counts = df['category'].value_counts()
    axes[0].bar(category_counts.index, category_counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_title('Category Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c'],
        startangle=90
    )
    axes[1].set_title('Category Distribution (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'category_distribution.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_processed_data(df, output_dir):
    """Save processed data as CSV"""
    print("\nSaving processed data...")
    
    output_file = output_dir / 'processed_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Saved: {output_file}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return output_file


def main():
    """Main execution"""
    print("="*80)
    print("DATA PREPARATION - SPORTS VS POLITICS CLASSIFIER")
    print("="*80)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / 'data' / 'raw' / 'News_Category_Dataset_v3.json'
    output_dir = base_dir / 'data' / 'processed'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not json_path.exists():
        print(f"ERROR: Dataset not found at {json_path}")
        print("Please download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        return False
    
    # Load and filter data
    articles = load_and_filter_data(json_path)
    
    # Create DataFrame
    df = create_dataframe(articles)
    
    # Encode labels
    df, label_mapping = encode_labels(df)
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Save processed data
    save_processed_data(df, output_dir)
    
    print("\n" + "="*80)
    print("✓ Data Preparation Complete")
    print("="*80)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
