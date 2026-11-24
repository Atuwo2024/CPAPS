
import os
import pandas as pd
from app import get_connection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = None

try:
    print("\n========== CPAPS PREDICTION DATA PREPARATION ==========")

    conn = get_connection()

    query = """
        SELECT
            county,
            area_type,
            day_of_week,
            hour_of_day,
            month_number,
            year_number,
            crime_type,
            crime_category,
            severity_level,
            property_loss_ksh,
            injuries_reported,
            fatalities
        FROM v_crime_report
    """

    df = pd.read_sql(query, conn)

    print("\nRecords retrieved:", len(df))

    # Standardise column names and remove incomplete records
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "county",
        "area_type",
        "day_of_week",
        "hour_of_day",
        "month_number",
        "year_number",
        "crime_type",
        "crime_category",
        "severity_level",
        "property_loss_ksh",
        "injuries_reported",
        "fatalities"
    ]

    df = df[required_columns]

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    df = df.dropna(subset=required_columns)

    # Convert numeric fields
    numeric_columns = [
        "hour_of_day",
        "month_number",
        "year_number",
        "property_loss_ksh",
        "injuries_reported",
        "fatalities"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)

    df["injuries_reported"] = df["injuries_reported"].astype(int)
    df["fatalities"] = df["fatalities"].astype(int)

    output_path = os.path.join(
        OUTPUT_DIR,
        "prediction_training_data.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n========== PREPARATION RESULTS ==========")
    print("Final records:", len(df))
    print("Columns:", list(df.columns))

    print("\nCrime type distribution:")
    print(df["crime_type"].value_counts())

    print("\nSeverity distribution:")
    print(df["severity_level"].value_counts())

    print("\nInjuries reported distribution:")
    print(df["injuries_reported"].value_counts())

    print("\nFatalities distribution:")
    print(df["fatalities"].describe())

    print("\nProperty loss distribution:")
    print(df["property_loss_ksh"].describe())

    print("\nDataset saved successfully:")
    print(output_path)

    print("\n========== PREPARATION COMPLETE ==========")

except Exception as e:
    print("\nERROR:", e)

finally:
    if conn is not None and conn.is_connected():
        conn.close()
