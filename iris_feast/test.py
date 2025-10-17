import pandas as pd
import feast
from joblib import load
import unittest


class TestModel(unittest.TestCase):
    def __init__(self):
        # Load model
        self.model = load("iris_model.bin")

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo/")

    def test_sample1(self):
        # Read features from Feast
        iris_ids = [1002]
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
        
        iris_features["prediction"] = self.model.predict(X)
        # Print
        encoder_dic = {
            1001: 'versicolor',
            1002: 'setosa',
            1003: 'virginica'
        }

        self.assertEqual(
            encoder_dic[iris_features.loc[0, "prediction"]],
            'setosa',
            "Prediction class is wrong"
        )


if __name__ == "__main__":
    unittest.main()
