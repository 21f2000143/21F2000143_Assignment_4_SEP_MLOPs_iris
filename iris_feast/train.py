import feast
from joblib import dump
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

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
train_Y = features_df.loc[:, target]
train_X = features_df.drop(columns=["iris_id"])
print("Passed during training")
print(train_X.info())
reg.fit(train_X, train_Y)

# Save model
dump(reg, "iris_model.bin")