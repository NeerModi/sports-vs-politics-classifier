"""
Sports vs Politics Classifier - Main Package
"""

__version__ = "1.0.0"
__author__ = "NLU Assignment Team"
__description__ = "Binary classification of news articles into Sports and Politics categories"

from .preprocess import clean_text, preprocess_batch

__all__ = ['clean_text', 'preprocess_batch']
