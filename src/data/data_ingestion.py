import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# --- Logging Configuration ---
# Ensure the logs folder exists
os.makedirs('logs', exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Changed to INFO to reduce console noise

file_handler = logging.FileHandler('logs/data_ingestion.log') # Save logs in a dedicated folder
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Dynamic path to params.yaml
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        params = load_params(params_path=params_path)
        
        # Extract parameters specifically for data_ingestion
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_url = params['data_ingestion']['data_url']
        
        # Load data
        df = load_data(data_url=data_url)
        
        # Split the data
        # We split early to prevent data leakage (test set is locked away immediately)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Save the split datasets
        data_path = os.path.join(curr_dir, '..', '..', 'data')
        save_data(train_data, test_data, data_path=data_path)
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    