
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cleaned_file = os.path.join(
    BASE_DIR, "ml_outputs", "cleaned_crime_data.csv"
)

monthly_file = os.path.join(
    BASE_DIR, "ml_outputs", "monthly_crime_patterns.csv"
)

cleaned = pd.read_csv(cleaned_file)
monthly = pd.read_csv(monthly_file)

cleaned["incident_date"] = pd.to_datetime(
    cleaned["incident_date"], errors="coerce"
)

monthly["month"] = pd.to_datetime(
    monthly["month"], errors="coerce"
)

print("\n========== VERIFYING ORIGINAL CRIME RECORDS ==========")

print("\n1. ORIGINAL CLEANED DATASET")
print("Total records:", len(cleaned))
print("Earliest incident:", cleaned["incident_date"].min())
print("Latest incident:", cleaned["incident_date"].max())

print("\n2. ORIGINAL RECORDS BEFORE MARCH 2020")

early_records = cleaned[
    cleaned["incident_date"] < "2020-03-01"
]

print("Number of records:", len(early_records))

if len(early_records) > 0:
    print(
        early_records[
            ["case_number", "incident_date", "county",
             "sub_county", "ward"]
        ].to_string(index=False)
    )
else:
    print("No original records exist before March 2020.")

print("\n3. JANUARY AND FEBRUARY 2020 IN MONTHLY DATA")

early_monthly = monthly[
    monthly["month"] < "2020-03-01"
]

print(
    early_monthly[
        ["county", "sub_county", "ward", "month", "total_cases"]
    ].to_string(index=False)
)

print("\n4. COMPARE ORIGINAL RECORDS WITH MONTHLY DATA")

if len(early_records) == 0:
    print(
        "WARNING: Monthly data contains January/February "
        "counts, but original records contain none."
    )
else:
    print(
        "Original records exist before March 2020. "
        "Check the dates and location counts above."
    )

print("\n========== VERIFICATION COMPLETE ==========")
