import feast
from joblib import dump
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from datetime import date

# Load entity dataframe with iris_id and timestamps
entity_df = pd.read_parquet("feature_repo/data/iris_stats.parquet")[["iris_id", "event_timestamp"]]
entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

# Connect to your local feature store
store = feast.FeatureStore(repo_path="feature_repo/")

# Fetch historical features
features_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_stats_source:sepal_length",
        "iris_stats_source:sepal_width",
        "iris_stats_source:petal_length",
        "iris_stats_source:petal_width",
    ]
).to_df()
print("Printing data obtained from the feast store")
print(features_df.head())
print(features_df.info())

# Train model
target = "iris_id"

reg = LogisticRegression(max_iter=200)
features_df = features_df.drop(columns=["event_timestamp"])
print("Splitting data into train and test sets")
train, test = train_test_split(features_df, test_size = 0.4, stratify = features_df['iris_id'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.iris_id
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.iris_id

print("Training model now")
print(X_train.info())
reg.fit(X_train, y_train)

print("Evaluating model now")
y_pred = reg.predict(X_test)
val_accuracy = metrics.accuracy_score(y_test, y_pred)
y_train_pred = reg.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)

# compute log loss (training and validation)
train_loss = metrics.log_loss(y_train, reg.predict_proba(X_train))
val_loss = metrics.log_loss(y_test, reg.predict_proba(X_test))

print(f"Training accuracy: {train_accuracy}")
print(f"Training loss: {train_loss}")
print(f"Validation accuracy: {val_accuracy}")
print(f"Validation loss: {val_loss}")

today = date.today().isoformat()  # YYYY-MM-DD

# Append metrics to the same CSV file with a date column (date only, no time)
with open("metrics.csv", "a") as f:
    f.write("\n")  # separator row
    f.write("date,train_accuracy,train_loss,val_accuracy,val_loss\n")
    f.write(f"{today},{train_accuracy},{train_loss},{val_accuracy},{val_loss}\n")

# Save model
dump(reg, "iris_model.bin")