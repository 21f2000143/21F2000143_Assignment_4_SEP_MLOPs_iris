import pandas as pd

iris_df = pd.read_csv('iris_data_adapted_for_feast.csv')

print(iris_df.head())

print("Converting and saving it to desired location")

iris_df.to_parquet('iris_feast/feature_repo/data/iris_stats.parquet', index=False)

print('Done!')