import pandas as pd
import feast
from joblib import load

# # Connect to your local feature store
# store = feast.FeatureStore(repo_path="feature_repo/")

# # Predict using online features
# online_df = store.get_online_features(
#     features=[
#         "iris_stats_source:sepal_length",
#         "iris_stats_source:sepal_width",
#         "iris_stats_source:petal_length",
#         "iris_stats_source:petal_width",
#     ],
#     entity_rows=[
#         {"iris_id": 1001},
#         {"iris_id": 1002},
#         {"iris_id": 1003},
#     ]
# ).to_df()
# print(online_df.head())
# print(online_df.info())


# # Drop ID column and reorder to match training feature order
# X_live = online_df.drop(columns=["iris_id"])
# X_live = X_live[X.columns]  # Reorder columns to match training data

# # Predict
# live_preds = model.predict(X_live)
# encoder_dic = {
#     1001: 'versicolor',
#     1002: 'setosa',
#     1003: 'virginica'
# }

# for p in live_preds:
#     print(f"iris_id {p} ➝ predicted species: {encoder_dic[p]}")

class Iris_Classifier_Model:
    def __init__(self):
        # Load model
        self.model = load("iris_model.bin")

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo/")

    def predict(self, iris_ids):
        # Read features from Feast
        iris_features = self.fs.get_online_features(
            entity_rows=[{"iris_id": iris_id} for iris_id in iris_ids],
            features=[
                "iris_stats_source:sepal_length",
                "iris_stats_source:sepal_width",
                "iris_stats_source:petal_length",
                "iris_stats_source:petal_width",
            ],
        ).to_df()
        expected_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X = iris_features.drop(columns=["iris_id"])   # keep only features
        X = X[expected_cols]
        print("passed data info")
        print(X.info())
        # Make prediction
        
        iris_features["prediction"] = self.model.predict(X)

        # Choose best driver
        best_iris_id = iris_features["iris_id"].iloc[iris_features["prediction"].argmax()]
        
        # Print
        encoder_dic = {
            1001: 'versicolor',
            1002: 'setosa',
            1003: 'virginica'
        }

        for p in iris_features["prediction"]:
            print(f"iris_id {p} ➝ predicted species: {encoder_dic[p]}")

        # return best driver
        return best_iris_id


if __name__ == "__main__":
    iris_ids = [1001, 1002, 1003]
    model = Iris_Classifier_Model()
    best_iris = model.predict(iris_ids)
    print(best_iris)
    
    