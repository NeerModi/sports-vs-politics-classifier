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

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'\S+@\S+', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens 
              if word not in STOP_WORDS and len(word) > 2]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


def preprocess_batch(texts)
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
