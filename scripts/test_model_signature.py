import mlflow
import pytest
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient

@pytest.mark.parametrize("model_name, vectorizer_path", [
    ("LightGBM_Reddit", "models/tfidf_vectorizer.pkl"), 
])
def test_model_with_vectorizer(model_name, vectorizer_path):
    client = MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = str(max([int(v.version) for v in versions]))

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        vectorizer = joblib.load(vectorizer_path)

        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        
        # THE FIX: Create DataFrame and force column names to be '0', '1', '2', etc.
        input_df = pd.DataFrame(input_data.toarray())
        input_df.columns = [str(i) for i in range(input_df.shape[1])]

        prediction = model.predict(input_df)

        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
