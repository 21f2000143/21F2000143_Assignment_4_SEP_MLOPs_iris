# 🚀 MLflow Setup and Integration Guide

## 📘 Overview

This document provides step-by-step instructions to set up **MLflow**.

---

## 🧠 Objective

Integrate **MLflow** into the homework pipeline by:

* Introducing **hyperparameter tuning** in the training loop.
* Logging **experiment parameters**, **evaluation metrics**, and **models** using MLflow.
* Demonstrating comparison of experiments via **Metric Visualization** in the MLflow UI.
* Removing existing model logging dependency from **DVC**.
* Modifying the evaluation pipeline to **fetch the best/latest model** from the MLflow registry.
* *(Optional)* Integrating **CI** to utilize models from MLflow for sanity checks.

---

## 🧰 Step 1: Install and Start MLflow

1. Install MLflow:

   ```bash
   pip install mlflow
   ```
2. Start the MLflow Tracking Server:

   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 8100
   ```
   - vertexAI workbench
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 8100 --allowed-hosts "34.29.180.60:8100"
   ```

---

## 💻 Step 2: Access MLflow from Local Machine

Use the **external IP** of your instance and the configured port to access MLflow:

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

