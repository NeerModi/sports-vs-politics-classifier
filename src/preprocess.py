"""
Text preprocessing module for sports vs politics classification.
Handles text cleaning, lowercasing, punctuation removal, stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
def _download_nltk_data():
    """Download required NLTK data"""
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource_name}: {e}")

_download_nltk_data()

# Initialize stopwords and lemmatizer
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess text.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation and special characters
    5. Tokenize
    6. Remove stopwords
    7. Lemmatize
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits, keep only alphabetic and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens 
              if word not in STOP_WORDS and len(word) > 2]
    
    # Join tokens back
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


def preprocess_batch(texts):
    """
    Preprocess a batch of texts.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        list: List of cleaned texts
    """
    return [clean_text(text) for text in texts]


if __name__ == "__main__":
    # Test the preprocessing function
    sample_texts = [
        "SPORTS: The team won the championship! Visit https://example.com for more.",
        "POLITICS: The government announced new policy changes today.",
        "Breaking News!!! Check @user for updates..."
    ]
    
    print("Text Preprocessing Module Test")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        cleaned = clean_text(text)
        print(f"\nOriginal Text {i}:\n{text}")
        print(f"\nCleaned Text {i}:\n{cleaned}")
        print("-" * 80)
