import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# --- Logging Configuration ---
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/data_preprocessing.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- NLTK Setup ---
def download_nltk_resources():
    """Download necessary NLTK data safely."""
    resources = ['wordnet', 'stopwords', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

# --- Preprocessing Functions ---

def preprocess_text(text):
    """
    Applies NLP preprocessing: Lowercase -> Strip -> Remove Punctuation -> Remove Stopwords -> Lemmatize
    """
    try:
        if not isinstance(text, str):
            return ""

        # 1. Convert to lowercase
        text = text.lower()

        # 2. Remove newlines and extra spaces
        text = re.sub(r'\n', ' ', text)
        text = text.strip()

        # 3. Remove Punctuation
        # We remove ALL punctuation here to ensure tokens are clean for Lemmatization.
        # If you strictly want to keep '!' for sentiment, you need a Tokenizer, not .split()
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 4. Remove Stopwords
        # Custom list: we keep negation words because they are vital for Sentiment Analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]

        # 5. Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        return ' '.join(lemmatized_words)

    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return text

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the basic cleaning (previously in ingestion) and applies text preprocessing.
    """
    try:
        initial_shape = df.shape
        
        # --- 1. Basic Cleaning (Restored from Data Ingestion) ---
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Ensure the text column is string
        # Assuming the column name is 'clean_comment' based on your input
        if 'clean_comment' in df.columns:
            # Filter empty strings (rows that are just whitespace)
            df = df[df['clean_comment'].astype(str).str.strip() != '']
        else:
            raise KeyError("Column 'clean_comment' not found in dataset")

        logger.debug(f"Basic cleaning dropped {initial_shape[0] - df.shape[0]} rows (NaNs/Duplicates/Empty)")

        # --- 2. Text Normalization ---
        df['clean_comment'] = df['clean_comment'].apply(preprocess_text)
        
        # Filter again just in case preprocessing resulted in empty strings
        df = df[df['clean_comment'] != '']
        
        logger.debug('Text normalization completed successfully')
        return df

    except Exception as e:
        logger.error(f"Error during data normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets to the interim folder."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.info(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        # Define paths relative to this script
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_path = os.path.join(curr_dir, '..', '..', 'data', 'raw')
        data_root_path = os.path.join(curr_dir, '..', '..', 'data')

        logger.info("Starting data preprocessing...")

        # Load Raw Data
        train_path = os.path.join(raw_data_path, 'train.csv')
        test_path = os.path.join(raw_data_path, 'test.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Raw data not found at {raw_data_path}. Run data_ingestion.py first.")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logger.debug(f"Loaded Raw Data: Train {train_data.shape}, Test {test_data.shape}")

        # Normalize (Clean + Preprocess)
        train_processed = normalize_data(train_data)
        test_processed = normalize_data(test_data)

        # Save to 'data/interim'
        save_data(train_processed, test_processed, data_path=data_root_path)

    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    