import os
import yaml
import logging
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Logging Setup ---
logger = logging.getLogger('common_utils')

# --- NLTK Resource Check ---
def ensure_nltk_resources():
    """Ensure required NLTK data is downloaded."""
    resources = ['wordnet', 'stopwords', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

ensure_nltk_resources()

# --- Global Initialization (Speed Optimization) ---
# Initialize ONCE, not inside the function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        logger.error(f"Error loading params from {params_path}: {e}")
        raise

def preprocess_text(text):
    """
    SHARED LOGIC: Applies NLP preprocessing.
    """
    try:
        if not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove Newlines
        text = re.sub(r'\n', ' ', text)

        # 3. Replace Punctuation with Space (Fixes "Good!Bad" -> "Good Bad")
        # We use regex to replace anything that is NOT a word or space with a space
        text = re.sub(r'[^\w\s]', ' ', text)

        # 4. Tokenize & Remove Stopwords
        # .split() handles multiple spaces automatically (e.g., "Good   Bad" -> ["Good", "Bad"])
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]

        # 5. Lemmatization
        # Uses the global lemmatizer object
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        return ' '.join(lemmatized_words)

    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return text