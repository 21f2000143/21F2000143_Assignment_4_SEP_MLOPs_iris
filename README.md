# 🚀 MLflow Setup and Integration Guide

## 📘 Overview

This document provides step-by-step instructions to set up Deployment of ml pipeline.

---

## 🧠 Objective

Deploy the ml pipeline using fastapi, docker, kubernetes engine, and CD using github actions
---

## 🧰 Step 1: Install and Start MLflow

## Steps to Run the Iris Classifier API

- install fastapi
```bash
pip install fastapi
```
or just install from requirements.txt
```bash
pip install -r requirements.txt
```  
Fire up Uvicorn for the ASGI app  
   ```bash
   uvicorn iris_fastapi:app --reload --host 0.0.0.0
   ```

- Test using curl
```bash
curl -X 'POST' 'http://34.86.74.149:8000/predict/' \
     -H 'Content-Type: application/json' \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## 💻 Step 2: Containerization

Build the Docker image:
```bash
docker build -t iris-app:latest .
```

- Create the container and start
```bash
docker run -d --name iris-model -p 8100:8100 iris-app:latest
```
- Start the existing container
```bash
docker start iris-model
```
```docker run -d --name musicapp -p 127.0.0.1:3000:3000 getting-started```
## To start an existing container
```docker start musicapp```

```
http://<external-ip>:8100
```

---

## 📔 Step 3: Integrate MLflow into Your Pipeline

1. Import MLflow in your training script:

   ```python
    # Setup mlflow tracking
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from pprint import pprint

    # Check if MLflow server is running and list experiments
    mlflow.set_tracking_uri("http://127.0.0.1:8100")
    client = MlflowClient(mlflow.get_tracking_uri())
    all_experiments = client.search_experiments()
    print("Existing experiments:")
    pprint([exp.name for exp in all_experiments])
    print("The tracking URI is set to:", mlflow.get_tracking_uri())

    print("setting the experiment to 'Iris_Classification'")
    mlflow.set_experiment("Iris_Classification")


   with mlflow.start_run(run_name="LogReg_C1.0"):
        mlflow.log_params(params)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.set_tag("model_name", "Iris_Classification")

        signature = mlflow.models.infer_signature(X_train, reg.predict(X_train))
        
        model_info = mlflow.sklearn.log_model(
            sk_model=reg,
            name="iris_model",
            signature=signature,
            registered_model_name="Iris_Classification_Model"
        )
   ```
2. Modify your pipeline to include hyperparameter tuning loops and log parameters/metrics accordingly.

---

## 🧮 Step 4: Compare Experiments

Try out different runs `run_name="LogReg_C0.1"`, `run_name="LogReg_C1.0"`, etc., with varying hyperparameters and log them using MLflow. Compare the results in the MLflow UI.

---

## 🧹 Step 5: Remove DVC Dependency

Remove any existing model logging or versioning code related to DVC from your training and evaluation scripts.

```mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 8100
```

```python
# evaluation of mlflow latest model
import mlflow
from mlflow import MlflowClient

# add this nippet of code to the test module
mlflow.set_tracking_uri("http://0.0.0.0:8100")
model_name = "Iris_Classification_Model"
client = MlflowClient()
latest_versions = client.get_latest_versions(model_name)
latest_version = max([int(v.version) for v in latest_versions])
print(f"Testing model: {model_name}, version: {latest_version}")
self.model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{latest_version}")
```

---

## 🔄 Step 6: Adding it to ci

Modify your evaluation pipeline:

```yml
- name: Start MLflow server (optional for testing)
    run: |
    mlflow server --backend-store-uri sqlite:///mlflow.db \
                    --default-artifact-root ./mlruns \
                    --host 127.0.0.1 --port 8100 &
    sleep 20  # wait for server to start
```

