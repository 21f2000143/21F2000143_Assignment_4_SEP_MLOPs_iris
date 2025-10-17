from feast import FeatureStore
from datetime import datetime, timedelta

# Connect to your local feature store
store = FeatureStore(repo_path="feature_repo/")

store.materialize(
    start_date=datetime.utcnow() - timedelta(days=180),
    end_date=datetime.utcnow()
)

# List feature views
print(store.list_feature_views())