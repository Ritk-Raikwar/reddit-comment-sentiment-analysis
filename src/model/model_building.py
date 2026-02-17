import numpy as np
import pandas as pd
import os
import logging
import sys
import joblib
import lightgbm as lgb

# --- Add Project Root to Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.common_text_preprocess import load_params

# --- Logging Configuration ---
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/model_building.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_model(X_train, y_train, model_params):
    """
    Trains the LightGBM model with specific parameters.
    """
    try:
        logger.info("Initializing LightGBM model...")
        
        # Handle random_state safety
        seed = model_params.get('random_state')
        if seed is None:
            seed = 42 
            logger.warning("Random state is empty in params.yaml. Using default 42.")

        # Initialize the model
        clf = lgb.LGBMClassifier(
            learning_rate=model_params['learning_rate'],
            n_estimators=model_params['n_estimators'],
            num_leaves=model_params['num_leaves'],
            class_weight=model_params['class_weight'],
            objective=model_params['objective'], 
            random_state=seed,
            n_jobs=-1 
        )
        
        # Train
        logger.info("Training started...")
        clf.fit(X_train, y_train)
        logger.info("Training completed.")
        
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def main():
    try:
        # 1. Load Paths and Params
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to params.yaml (Root)
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        params = load_params(params_path)
        
        # Path to processed data (Root/data/processed)
        processed_data_path = os.path.join(curr_dir, '..', '..', 'data', 'processed')
        
        # Path to save artifacts (Root/models)
        # Note: We go up two levels from src/model/ to reach root
        models_path = os.path.join(curr_dir, '..', '..', 'models')
        os.makedirs(models_path, exist_ok=True)

        target_col = params['data_preprocessing']['target_column']
        model_params = params['model_building']

        logger.info("Loading processed training data...")

        # 2. Load Data (TRAIN ONLY)
        train_df = pd.read_csv(os.path.join(processed_data_path, 'train_tfidf.csv'))
        
        # 3. Separate Features (X) and Target (y)
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # 4. Train Model
        clf = train_model(X_train, y_train, model_params)

        # 5. Save Model to Root/models/
        save_path = os.path.join(models_path, 'lightgbm_model.pkl')
        joblib.dump(clf, save_path)
        logger.info(f"Model saved successfully to {save_path}")

    except Exception as e:
        logger.error(f"Failed to complete model building: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()