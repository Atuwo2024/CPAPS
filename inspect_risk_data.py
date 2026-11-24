
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "monthly_crime_patterns.csv"
)

df = pd.read_csv(FILE)

df["month"] = pd.to_datetime(
    df["month"],
    format="%Y-%m",
    errors="coerce"
)

df["total_cases"] = pd.to_numeric(
    df["total_cases"],
    errors="coerce"
)

locations = ["county", "sub_county", "ward"]

print("\n========== CPAPS RISK DATA INVESTIGATION ==========")

print("\n1. DATASET OVERVIEW")
print("Total rows:", len(df))
print("Unique locations:", df[locations].drop_duplicates().shape[0])
print("First month:", df["month"].min())
print("Last month:", df["month"].max())

print("\n2. MONTHLY CRIME COUNT DISTRIBUTION")
print(df["total_cases"].value_counts().sort_index())

print("\n3. ZERO AND NON-ZERO MONTHS")
print("Zero-crime months:", (df["total_cases"] == 0).sum())
print("Non-zero months:", (df["total_cases"] > 0).sum())
print(
    "Percentage of zero months:",
    round((df["total_cases"] == 0).mean() * 100, 2),
    "%"
)

print("\n4. CRIME COUNT SUMMARY")
print(df["total_cases"].describe())

print("\n5. DISTRIBUTION BY YEAR")
df["year"] = df["month"].dt.year

print(
    df.groupby("year")["total_cases"]
    .agg(["count", "sum", "mean", "min", "max"])
)

print("\n6. MONTHLY COUNTS BY LOCATION")

for location, group in df.groupby(locations):
    print("\nLocation:", location)
    print("Months:", len(group))
    print("Total cases:", group["total_cases"].sum())
    print("Zero months:", (group["total_cases"] == 0).sum())
    print("Average cases:", round(group["total_cases"].mean(), 2))
    print("Maximum monthly cases:", group["total_cases"].max())

print("\n7. FIRST TWO MONTHS OF 2020")

early = df[df["month"] < "2020-03-01"]

print(early[locations + ["month", "total_cases"]].to_string(index=False))

print("\n8. RISK THRESHOLD CHECK")

print("33rd percentile:", df["total_cases"].quantile(0.33))
print("67th percentile:", df["total_cases"].quantile(0.67))

print("\n========== INVESTIGATION COMPLETE ==========")
