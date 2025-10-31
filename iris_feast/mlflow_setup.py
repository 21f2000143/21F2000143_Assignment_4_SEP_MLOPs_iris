# Setup mlflow tracking
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint

# Check if MLflow server is running and list experiments
mlflow.set_tracking_uri("http://0.0.0.0:8100")
client = MlflowClient(mlflow.get_tracking_uri())
all_experiments = client.search_experiments()
print("Existing experiments:")
pprint([exp.name for exp in all_experiments])
print("The tracking URI is set to:", mlflow.get_tracking_uri())

print("setting the experiment to 'Iris_Classification'")
mlflow.set_experiment("Iris_Classification: Mlflow")
