import pandas as pd
import numpy as np
from datetime import date
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------
# Load CSV
# ---------------------------------------------
df = pd.read_csv("data/raw/iris.csv")
df_test = pd.read_csv("data/v2/data.csv")

# Features and labels
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = df["species"].values

X_test = df_test[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y_test = df_test["species"].values

# Encode string labels -> integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)   # setosa → 0, versicolor → 1, virginica → 2 (if present)
y_test = label_encoder.transform(y_test)
# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


def poison_labels(y, percent, num_classes):
    y = y.copy()
    n = len(y)
    k = int(n * percent/100)

    indices = np.random.choice(n, k, replace=False)

    for i in indices:
        original = y[i]
        new_label = np.random.choice([x for x in range(num_classes) if x != original])
        y[i] = new_label

    return y


# ---------------------------------------------
# MLP Model
# ---------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(np.unique(y)))  # changes based on number of species
        )

    def forward(self, x):
        return self.net(x)


criterion = nn.CrossEntropyLoss()

epochs = 5

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

# ---------------------------------------------
# MLflow Tracking
# ---------------------------------------------
from mlflow_setup import mlflow
from mlflow.models import infer_signature

# Poisoning levels to test
poison_levels = [5, 10, 50]
num_classes = len(np.unique(y))

for p_level in poison_levels:
    with mlflow.start_run(run_name=f"MLP_Iris_Poisoned_{p_level}pct"):
        mlflow.log_param("poison_level", p_level)
        mlflow.log_param("model", "MLP")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        y_train_poisoned = poison_labels(y, p_level, num_classes)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_poisoned, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        model = MLP()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            # ---- Training ----
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

            train_acc = accuracy(logits, y_train_t)

            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                val_logits = model(X_test_t)
                val_loss = criterion(val_logits, y_test_t)
                val_acc = accuracy(val_logits, y_test_t)

            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

            # Log metrics per epoch
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Save model
        # Infer signature
        # Ensure input example is float32 to match the model
        input_example = X_train[:1].astype(np.float32)

        signature = infer_signature(input_example, model(torch.tensor(input_example)).detach().numpy())

        mlflow.pytorch.log_model(
            model,
            name="mlp_iris_model",
            signature=signature,
            input_example=input_example
        )

        dump(scaler, "scaler.bin")
        dump(label_encoder, "label_encoder.bin")

    torch.save(model.state_dict(), "iris_mlp_model.pt")
    print("Training complete.")
