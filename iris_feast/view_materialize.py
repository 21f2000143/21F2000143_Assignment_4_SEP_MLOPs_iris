from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Try querying features for flower_id 0, 1, 2
features = store.get_online_features(
    features=[
        "iris_stats_source:sepal_length",
        "iris_stats_source:sepal_width",
        "iris_stats_source:petal_length",
        "iris_stats_source:petal_width"
    ],
    entity_rows=[
        {"iris_id": 1001},
        {"iris_id": 1002},
        {"iris_id": 1003},        
    ]
).to_df()

print(features.info())
print(features)