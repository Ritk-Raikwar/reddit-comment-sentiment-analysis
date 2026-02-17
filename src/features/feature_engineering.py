import pandas as pd
import os
import sys
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Add Project Root to Path ---
# This must happen BEFORE importing from src.utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.common_text_preprocess import load_params

# --- Logging Setup ---
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/feature_engineering.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    try:
        # 1. Load Params
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        params = load_params(params_path)

        # Get configurations
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])
        text_col = params['data_preprocessing']['text_column']
        target_col = params['data_preprocessing']['target_column']
        
        # Paths
        data_path = os.path.join(curr_dir, '..', '..', 'data')
        interim_path = os.path.join(data_path, 'interim')
        processed_path = os.path.join(data_path, 'processed')
        models_path = os.path.join(curr_dir, '..', '..', 'models')
        
        os.makedirs(processed_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        logger.info("Starting feature engineering...")

        # 2. Load Data
        train_df = pd.read_csv(os.path.join(interim_path, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(interim_path, 'test_processed.csv'))
        
        # Fill NaNs just in case (TF-IDF cannot handle NaN)
        train_df[text_col] = train_df[text_col].fillna('')
        test_df[text_col] = test_df[text_col].fillna('')

        # 3. Initialize and Fit TF-IDF Vectorizer
        logger.debug(f"Fitting TF-IDF with max_features={max_features}, ngram_range={ngram_range}")
        
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        # Fit ONLY on Train data to prevent data leakage
        X_train_tfidf = vectorizer.fit_transform(train_df[text_col])
        X_test_tfidf = vectorizer.transform(test_df[text_col]) # Transform test data

        # 4. Save the Vectorizer (CRITICAL for Inference)
        joblib.dump(vectorizer, os.path.join(models_path, 'tfidf_vectorizer.pkl'))
        logger.info("TF-IDF Vectorizer saved successfully.")

        # 5. Save Transformed Data
        # We convert the sparse matrix to a DataFrame for saving as CSV
        # Note: For very large datasets, saving as CSV is inefficient. Consider .npz or .parquet.
        train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
        test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray())
        
        # Add the target column back so the model knows what to predict
        train_tfidf_df[target_col] = train_df[target_col].values
        test_tfidf_df[target_col] = test_df[target_col].values

        train_tfidf_df.to_csv(os.path.join(processed_path, 'train_tfidf.csv'), index=False)
        test_tfidf_df.to_csv(os.path.join(processed_path, 'test_tfidf.csv'), index=False)
        
        logger.info("Feature engineering completed. Processed data saved.")

    except Exception as e:
        logger.error(f"Failed to complete feature engineering: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    