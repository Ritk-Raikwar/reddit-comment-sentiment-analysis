import numpy as np
import pandas as pd
import os
import logging
import sys
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Add Project Root to Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.common_text_preprocess import load_params

# --- Logging Setup ---
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/model_evaluation.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def eval_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    Since we have 3 classes (-1, 0, 1), we use 'weighted' average.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def main():
    try:
        # 1. Load Paths
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        params = load_params(params_path)
        
        processed_data_path = os.path.join(curr_dir, '..', '..', 'data', 'processed')
        models_path = os.path.join(curr_dir, '..', '..', 'models')
        
        target_col = params['data_preprocessing']['target_column']

        logger.info("Loading model and test data...")

        # 2. Load Model & Test Data
        model_path = os.path.join(models_path, 'lightgbm_model.pkl')
        clf = joblib.load(model_path)

        test_df = pd.read_csv(os.path.join(processed_data_path, 'test_tfidf.csv'))
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 3. Predict
        predicted_qualities = clf.predict(X_test)

        # 4. Calculate Metrics
        accuracy, precision, recall, f1 = eval_metrics(y_test, predicted_qualities)

        logger.info(f"Evaluation Metrics - Accuracy: {accuracy}, F1: {f1}")

        # 5. Save Metrics to JSON (Crucial for DVC!)
        metrics_file = "metrics.json"
        
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        with open(metrics_file, "w") as f:
            json.dump(scores, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()