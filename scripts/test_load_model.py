import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

@pytest.mark.parametrize("model_name", ["LightGBM_Reddit"])
def test_load_latest_model(model_name):
    client = MlflowClient()
    
    try:
        # Fetch all versions and find the highest version number automatically
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = str(max([int(v.version) for v in versions]))
        
        # Load the absolute latest model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
