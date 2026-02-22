import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    client = MlflowClient()
    model_name = "LightGBM_Reddit"

    # Find the latest version we just tested
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = str(max([int(v.version) for v in versions]))

    # Set the 'champion' alias to this version (Modern MLflow Production standard)
    client.set_registered_model_alias(model_name, "champion", latest_version)
    print(f"Model version {latest_version} successfully promoted to Champion (Production)!")

if __name__ == "__main__":
    promote_model()
    