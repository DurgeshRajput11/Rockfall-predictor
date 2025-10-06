import pandas as pd
import yaml

# Load train.csv and print columns
train_path = r"Artifacts\10_06_2025_00_10_37\data_ingestion\ingested\train.csv"
df = pd.read_csv(train_path)
print("train.csv columns:", list(df.columns))

# Load schema.yaml and print columns
with open(r"data_schema\schema.yaml", "r") as f:
    schema = yaml.safe_load(f)
schema_columns = [col['name'] for col in schema['columns']]
print("schema.yaml columns:", schema_columns)
