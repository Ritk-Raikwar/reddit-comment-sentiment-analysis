import numpy as np
import pandas as pd
import os
import sys
import logging

# --- Add Project Root to Path ---
# This must happen BEFORE importing from src.utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import safely
from src.utils.common_text_preprocess import preprocess_text, load_params

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


def normalize_data(df: pd.DataFrame, text_col: str, target_col: str) -> pd.DataFrame:
    """
    Performs basic cleaning and text preprocessing.
    Uses .copy() to prevent SettingWithCopyWarning.
    """
    try:
        initial_shape = df.shape
        
        # --- 1. Basic Cleaning ---
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Verify columns exist
        if text_col not in df.columns:
            raise KeyError(f"Text column '{text_col}' not found. Check params.yaml.")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found. Check params.yaml.")

        # --- Fix for SettingWithCopyWarning ---
        # Filter empty strings and create a deep COPY of the dataframe
        # This tells Pandas: "This is a new object, not a slice of the old one."
        df = df[df[text_col].astype(str).str.strip() != ''].copy()

        logger.debug(f"Basic cleaning dropped {initial_shape[0] - df.shape[0]} rows (NaNs/Duplicates/Empty)")

        # --- 2. Text Normalization ---
        # Apply the shared preprocessing function
        df[text_col] = df[text_col].apply(preprocess_text)
        
        # Filter again just in case preprocessing resulted in empty strings
        df = df[df[text_col] != ''].copy()
        
        # Keep only the relevant columns to save space
        df = df[[text_col, target_col]]
        
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
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        raw_data_path = os.path.join(curr_dir, '..', '..', 'data', 'raw')
        data_root_path = os.path.join(curr_dir, '..', '..', 'data')

        logger.info("Starting data preprocessing...")

        # 1. Load Params
        params = load_params(params_path)
        text_col = params['data_preprocessing']['text_column']
        target_col = params['data_preprocessing']['target_column']

        # 2. Load Raw Data
        train_path = os.path.join(raw_data_path, 'train.csv')
        test_path = os.path.join(raw_data_path, 'test.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Raw data not found at {raw_data_path}. Run data_ingestion.py first.")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logger.debug(f"Loaded Raw Data: Train {train_data.shape}, Test {test_data.shape}")

        # 3. Normalize (Clean + Preprocess)
        # We pass the column names from params.yaml
        train_processed = normalize_data(train_data, text_col, target_col)
        test_processed = normalize_data(test_data, text_col, target_col)

        # 4. Save to 'data/interim'
        save_data(train_processed, test_processed, data_path=data_root_path)

    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    