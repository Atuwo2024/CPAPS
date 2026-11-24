
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "cleaned_crime_data.csv"
)

df = pd.read_csv(FILE)

print("\n========== CPAPS PREDICTION DATA INSPECTION ==========")

print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. AVAILABLE COLUMNS")
for column in df.columns:
    print("-", column)

print("\n3. CHECK PREDICTION TARGETS")

possible_targets = [
    "crime_type",
    "crime_category",
    "severity_level",
    "property_loss",
    "property_loss_ksh",
    "injuries",
    "injuries_reported",
    "fatalities"
]

for column in possible_targets:
    if column in df.columns:
        print(f"\n{column}")
        print("Missing values:", df[column].isna().sum())
        print("Unique values:", df[column].nunique())
        print("Sample values:")
        print(df[column].value_counts().head(10))
    else:
        print(f"\n{column}: NOT FOUND")

print("\n========== INSPECTION COMPLETE ==========")
