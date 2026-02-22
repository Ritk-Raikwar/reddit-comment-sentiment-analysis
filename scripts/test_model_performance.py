import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

@pytest.mark.parametrize("model_name, holdout_data_path, vectorizer_path", [
    ("LightGBM_Reddit", "data/interim/test_processed.csv", "models/tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, holdout_data_path, vectorizer_path):
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = str(max([int(v.version) for v in versions]))

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        vectorizer = joblib.load(vectorizer_path)

        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze() 
        y_holdout = holdout_data.iloc[:, -1]

        X_holdout_raw = X_holdout_raw.fillna("")

        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        
        # THE FIX: Create DataFrame and force column names to be '0', '1', '2', etc.
        X_holdout_tfidf_df = pd.DataFrame(X_holdout_tfidf.toarray())
        X_holdout_tfidf_df.columns = [str(i) for i in range(X_holdout_tfidf_df.shape[1])]

        y_pred_new = model.predict(X_holdout_tfidf_df)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        assert accuracy_new >= expected_accuracy, f'Accuracy should be at least {expected_accuracy}, got {accuracy_new}'
        assert precision_new >= expected_precision, f'Precision should be at least {expected_precision}, got {precision_new}'
        assert recall_new >= expected_recall, f'Recall should be at least {expected_recall}, got {recall_new}'
        assert f1_new >= expected_f1, f'F1 score should be at least {expected_f1}, got {f1_new}'

        print(f"Performance test passed for model '{model_name}' version {latest_version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")
