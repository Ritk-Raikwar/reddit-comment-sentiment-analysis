import numpy as np
import pandas as pd
import os
import logging
import sys
import joblib
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv  # <--- CRITICAL: Required to read .env file

# --- Add Project Root to Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.common_text_preprocess import load_params

# --- Logging Setup ---
os.makedirs('logs', exist_ok=True)
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
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def log_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        # 1. Load Environment Variables (DagsHub Credentials)
        load_dotenv()
        
        # 2. Load Paths
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(curr_dir, '..', '..', 'params.yaml')
        params = load_params(params_path)
        
        processed_data_path = os.path.join(curr_dir, '..', '..', 'data', 'processed')
        models_path = os.path.join(curr_dir, '..', '..', 'models')
        
        target_col = params['data_preprocessing']['target_column']
        
        # Check if DagsHub URI is present
        dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
        if dagshub_uri:
            logger.info(f"MLflow Tracking URI detected: {dagshub_uri}")
            mlflow.set_tracking_uri(dagshub_uri)
        else:
            logger.warning("MLFLOW_TRACKING_URI not found in .env file! Logging locally.")

        logger.info("Loading model and test data...")

        # 3. Load Model & Data
        model_path = os.path.join(models_path, 'lightgbm_model.pkl')
        vectorizer_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
        
        clf = joblib.load(model_path)
        test_df = pd.read_csv(os.path.join(processed_data_path, 'test_tfidf.csv'))
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 4. Predict
        predicted_qualities = clf.predict(X_test)

        # 5. Calculate Metrics
        accuracy, precision, recall, f1 = eval_metrics(y_test, predicted_qualities)
        
        # --- MLFLOW LOGGING START ---
        logger.info("Starting MLflow run...")
        mlflow.set_experiment("Reddit Sentiment Analysis")
        
        with mlflow.start_run():
            # A. Log Params
            mlflow.log_params(params['model_building'])
            mlflow.log_param("features_max_features", params['feature_engineering']['max_features'])
            
            # B. Log Metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            # C. Log Confusion Matrix Image
            cm_path = "confusion_matrix.png"
            log_confusion_matrix(y_test, predicted_qualities, cm_path)
            mlflow.log_artifact(cm_path)
            if os.path.exists(cm_path): os.remove(cm_path) # Clean up local image

            # D. Log Model with Signature
            # Infer signature (input schema)
            signature = infer_signature(X_test, predicted_qualities)
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Log model differently depending on if we are remote or local
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    clf, 
                    "model", 
                    registered_model_name="LightGBM_Reddit",
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(clf, "model", signature=signature)

            # E. Log Vectorizer as Artifact
            # This is critical to ensure we can preprocess new text later
            mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")
            
        logger.info("MLflow logging completed.")
        # --- MLFLOW LOGGING END ---

        # 6. Save Metrics to JSON (Keep this for DVC!)
        metrics_file = "metrics.json"
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        with open(metrics_file, "w") as f:
            json.dump(scores, f, indent=4)
        
        logger.info(f"Local metrics saved to {metrics_file}")

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()