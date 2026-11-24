
import os
import mysql.connector
import pandas as pd


OUTPUT_DIR = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. DATABASE CONNECTION
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="cpaps_db"
    )


# 2. LOAD CRIME DATA
def load_crime_data():

    connection = get_connection()

    query = """
        SELECT
            case_number,
            incident_date,
            incident_time,
            crime_type,
            crime_category,
            severity_level,
            county,
            sub_county,
            ward,
            area_type,
            latitude,
            longitude,
            day_of_week,
            month_number,
            year_number,
            hour_of_day
        FROM v_crime_report
    """

    try:
        df = pd.read_sql(query, connection)
        print("Crime records loaded:", len(df))
        return df

    finally:
        connection.close()


# 3. CLEAN DATA
def clean_data(df):

    df = df.copy()

    df["incident_date"] = pd.to_datetime(
        df["incident_date"], errors="coerce"
    )

    df = df.dropna(subset=["incident_date"])

    text_columns = [
        "crime_type",
        "crime_category",
        "severity_level",
        "county",
        "sub_county",
        "ward",
        "area_type"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    df["month"] = df["incident_date"].dt.to_period("M").astype(str)

    print("Records after date cleaning:", len(df))

    return df


# 4. MONTHLY CRIME PATTERNS BY LOCATION
def prepare_monthly_patterns(df):

    location_columns = ["county", "sub_county", "ward"]

    monthly = (
        df.groupby(location_columns + ["month"])
        .size()
        .reset_index(name="total_cases")
    )

    all_months = pd.period_range(
        start=df["incident_date"].min().to_period("M"),
        end=df["incident_date"].max().to_period("M"),
        freq="M"
    ).astype(str)

    location_groups = df[location_columns].drop_duplicates()

    complete_data = []

    for _, location in location_groups.iterrows():

        mask = pd.Series(True, index=monthly.index)

        for column in location_columns:
            mask &= monthly[column] == location[column]

        location_data = monthly.loc[mask, ["month", "total_cases"]]

        location_data = (
            location_data.set_index("month")
            .reindex(all_months, fill_value=0)
            .rename_axis("month")
            .reset_index()
        )

        for column in location_columns:
            location_data[column] = location[column]

        complete_data.append(location_data)

    result = pd.concat(complete_data, ignore_index=True)

    result = result[
        location_columns + ["month", "total_cases"]
    ]

    result.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_crime_patterns.csv"),
        index=False
    )

    print("Monthly location dataset saved.")


# 5. MONTHLY CRIME TYPE DISTRIBUTION
def prepare_crime_types(df):

    result = (
        df.groupby(["month", "crime_type"])
        .size()
        .reset_index(name="total_cases")
    )

    result.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_crime_types.csv"),
        index=False
    )

    print("Monthly crime type dataset saved.")


# 6. MONTHLY SEVERITY DISTRIBUTION
def prepare_severity(df):

    result = (
        df.groupby(["month", "severity_level"])
        .size()
        .reset_index(name="total_cases")
    )

    result.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_crime_severity.csv"),
        index=False
    )

    print("Monthly severity dataset saved.")


# 7. MAIN EXECUTION
if __name__ == "__main__":

    print("\n================================")
    print("CPAPS DATA PREPARATION STARTED")
    print("================================\n")

    try:
        crime_data = load_crime_data()

        crime_data = clean_data(crime_data)

        crime_data.to_csv(
            os.path.join(OUTPUT_DIR, "cleaned_crime_data.csv"),
            index=False
        )

        prepare_monthly_patterns(crime_data)
        prepare_crime_types(crime_data)
        prepare_severity(crime_data)

        print("\n================================")
        print("DATA PREPARATION COMPLETED")
        print("================================")

    except Exception as error:
        print("\nERROR:", error)
