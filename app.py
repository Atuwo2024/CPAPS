
from flask import Flask, render_template, jsonify, request
import mysql.connector
import pandas as pd
import joblib
import numpy as np
import os

from mysql.connector import Error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

app.config["JSON_SORT_KEYS"] = False



DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "database": "cpaps_db"
}


def get_connection():

    return mysql.connector.connect(**DB_CONFIG)



def query_db(sql, params=None):

    conn = None

    try:

        conn = get_connection()

        df = pd.read_sql_query(
            sql,
            conn,
            params=params
        )

        return df

    except Error as e:

        print("Database error:", e)
        raise

    except Exception as e:

        print("Query error:", e)
        raise

    finally:

        if conn and conn.is_connected():
            conn.close()



def dataframe_to_json(df):

    clean_df = df.astype(object).where(
        pd.notna(df),
        None
    )

    return clean_df.to_dict(orient="records")


def safe_int(value):

    if pd.isna(value):
        return 0

    return int(value)


def safe_float(value):

    if pd.isna(value):
        return 0.0

    return float(value)


def get_top_value(series):

    values = series.dropna()

    if values.empty:
        return "N/A"

    return str(values.mode().iloc[0])



MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs"
)


def load_model(filename):

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(model_path):

        print(f"WARNING: Model not found: {model_path}")

        return None

    try:

        model = joblib.load(model_path)

        print(f"Successfully loaded: {filename}")

        return model

    except Exception as e:

        print(f"Error loading {filename}: {e}")

        return None


rf_model = load_model(
    "random_forest_model.pkl"
)

lr_model = load_model(
    "logistic_regression_model.pkl"
)

lr_scaler = load_model(
    "lr_scaler.pkl"
)

dt_model = load_model(
    "decision_tree_model.pkl"
)



def models_available():

    return all([
        rf_model is not None,
        lr_model is not None,
        lr_scaler is not None,
        dt_model is not None
    ])



@app.route("/")
def dashboard():

    return render_template("dashboard.html")


@app.route("/records")
def records():

    return render_template("records.html")


@app.route("/predict")
def predict_page():

    return render_template("predict.html")


@app.route("/hotspots")
def hotspots():

    return render_template("hotspots.html")


@app.route("/map")
def crime_map():

    return render_template("crime_map.html")


@app.route("/api/stats")
def api_stats():

    try:

        df = query_db(
            "SELECT * FROM v_crime_report"
        )

        if df.empty:

            return jsonify({
                "total_incidents": 0,
                "open_cases": 0,
                "total_loss": "KES 0",
                "total_fatalities": 0,
                "top_county": "N/A",
                "top_crime": "N/A"
            })

        total_incidents = len(df)

        open_cases = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("open")
            .sum()
        )

        total_loss = pd.to_numeric(
            df["property_loss_ksh"],
            errors="coerce"
        ).fillna(0).sum()

        total_fatalities = pd.to_numeric(
            df["fatalities"],
            errors="coerce"
        ).fillna(0).sum()

        stats = {

            "total_incidents": int(total_incidents),

            "open_cases": int(open_cases),

            "total_loss": f"KES {total_loss:,.0f}",

            "total_fatalities": int(total_fatalities),

            "top_county": get_top_value(
                df["county"]
            ),

            "top_crime": get_top_value(
                df["crime_type"]
            )

        }

        return jsonify(stats)

    except Exception as e:

        print("Dashboard statistics error:", e)

        return jsonify({
            "error": "Unable to load dashboard statistics"
        }), 500


@app.route("/api/crime_by_type")
def api_crime_by_type():

    try:

        df = query_db("""
            SELECT
                crime_type,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY crime_type

            ORDER BY count DESC
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Crime type error:", e)

        return jsonify({
            "error": "Unable to load crime types"
        }), 500


@app.route("/api/crime_by_county")
def api_crime_by_county():

    try:

        df = query_db("""
            SELECT
                county,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY county

            ORDER BY count DESC
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("County error:", e)

        return jsonify({
            "error": "Unable to load county statistics"
        }), 500


@app.route("/api/crime_by_hour")
def api_crime_by_hour():

    try:

        df = query_db("""
            SELECT
                hour_of_day,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY hour_of_day

            ORDER BY hour_of_day
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Hourly crime error:", e)

        return jsonify({
            "error": "Unable to load hourly statistics"
        }), 500



@app.route("/api/monthly_trend")
def api_monthly_trend():

    try:

        df = query_db("""
            SELECT
                year_number,
                month_number,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY
                year_number,
                month_number

            ORDER BY
                year_number,
                month_number
        """)

        if df.empty:
            return jsonify([])

        df["year_number"] = pd.to_numeric(
            df["year_number"],
            errors="coerce"
        )

        df["month_number"] = pd.to_numeric(
            df["month_number"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["year_number", "month_number"]
        )

        df["label"] = (
            df["year_number"].astype(int).astype(str)
            + "-"
            + df["month_number"].astype(int).astype(str).str.zfill(2)
        )

        return jsonify(
            dataframe_to_json(
                df[["label", "count"]]
            )
        )

    except Exception as e:

        print("Monthly trend error:", e)

        return jsonify({
            "error": "Unable to load monthly trends"
        }), 500

@app.route("/api/severity")
def api_severity():

    try:

        df = query_db("""
            SELECT
                severity_level,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY severity_level

            ORDER BY count DESC
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Severity error:", e)

        return jsonify({
            "error": "Unable to load severity statistics"
        }), 500


@app.route("/api/crime_by_day")
def api_crime_by_day():

    try:

        df = query_db("""
            SELECT
                day_of_week,
                COUNT(*) AS count

            FROM v_crime_report

            GROUP BY day_of_week
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Crime by day error:", e)

        return jsonify({
            "error": "Unable to load daily statistics"
        }), 500


@app.route("/api/records")
def api_records():

    try:

        df = query_db("""
            SELECT
                case_number,
                incident_date,
                crime_type,
                crime_category,
                county,
                sub_county,
                severity_level,
                status,
                property_loss_ksh

            FROM v_crime_report

            ORDER BY incident_date DESC

            LIMIT 200
        """)

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Records error:", e)

        return jsonify({
            "error": "Unable to load crime records"
        }), 500


@app.route("/api/hotspots")
def api_hotspots():

    try:

        df = query_db(
            "SELECT * FROM v_hotspot_summary"
        )

        return jsonify(
            dataframe_to_json(df)
        )

    except Exception as e:

        print("Hotspots error:", e)

        return jsonify({
            "error": "Unable to load hotspot information"
        }), 500



def validate_prediction_data(data):

    if not isinstance(data, dict):

        raise ValueError(
            "A valid JSON object is required."
        )

    def get_numeric(
        key,
        default,
        minimum,
        maximum,
        is_float=False
    ):

        try:

            number = float(
                data.get(key, default)
            )

            if not np.isfinite(number):
                raise ValueError

            if not is_float and not number.is_integer():
                raise ValueError

            if not minimum <= number <= maximum:
                raise ValueError

            return number if is_float else int(number)

        except (TypeError, ValueError, OverflowError):

            raise ValueError(
                f"Invalid value for {key}. "
                f"Expected {minimum} to {maximum}."
            )

    return {

        "hour_of_day": get_numeric(
            "hour_of_day", 12, 0, 23
        ),

        "month_number": get_numeric(
            "month_number", 6, 1, 12
        ),

        "year_number": get_numeric(
            "year_number", 2024, 1900, 2100
        ),

        "property_loss": get_numeric(
            "property_loss", 0, 0, 1000000000, True
        ),

        "injuries": get_numeric(
            "injuries", 0, 0, 100000
        ),

        "fatalities": get_numeric(
            "fatalities", 0, 0, 100000
        )

    }

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        # Check ML models

        if not models_available():

            return jsonify({
                "error": "Machine learning models are unavailable. Check the ml_outputs folder."
            }), 503

        # Read request data

        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({
                "error": "Please submit valid JSON data."
            }), 400

        # Validate numerical inputs

        validated = validate_prediction_data(data)


        county_map = {
            "Nairobi": 0,
            "Mombasa": 1,
            "Kisumu": 2,
            "Nakuru": 3,
            "Machakos": 4
        }

        area_map = {
            "Urban": 2,
            "Suburban": 1,
            "Rural": 0
        }

        day_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

        severity_map = {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Critical": 3
        }

        county = str(
            data.get("county", "Nairobi")
        ).strip()

        area_type = str(
            data.get("area_type", "Urban")
        ).strip()

        day = str(
            data.get("day_of_week", "Monday")
        ).strip()

        severity = str(
            data.get("severity_level", "Medium")
        ).strip()

        if county not in county_map:
            raise ValueError("Invalid county selected.")

        if area_type not in area_map:
            raise ValueError("Invalid area type selected.")

        if day not in day_map:
            raise ValueError("Invalid day selected.")

        if severity not in severity_map:
            raise ValueError("Invalid severity selected.")


        features = np.array([[
            validated["hour_of_day"],
            validated["month_number"],
            validated["year_number"],
            county_map[county],
            area_map[area_type],
            day_map[day],
            severity_map[severity],
            validated["property_loss"],
            validated["injuries"],
            validated["fatalities"]
        ]], dtype=float)

        crime_types = [
            "Assault",
            "Burglary",
            "Disorderly Conduct",
            "Drug Possession",
            "Drug Trafficking",
            "Fraud/Scam",
            "Identity Theft",
            "Illegal Gathering",
            "Murder",
            "Rape/Sexual Assault",
            "Robbery with Violence",
            "Theft",
            "Vandalism",
            "Vehicle Theft"
        ]

        rf_pred = int(
            rf_model.predict(features)[0]
        )

        rf_proba = rf_model.predict_proba(
            features
        )[0]

        rf_classes = list(rf_model.classes_)

        if rf_pred not in rf_classes:

            raise ValueError(
                "Random Forest prediction class is invalid."
            )

        rf_confidence = float(
            max(rf_proba)
        )

        crime_type = (
            crime_types[rf_pred]
            if 0 <= rf_pred < len(crime_types)
            else "Unknown"
        )

        scaled_features = lr_scaler.transform(
            features
        )

        lr_pred = int(
            lr_model.predict(scaled_features)[0]
        )

        lr_proba = lr_model.predict_proba(
            scaled_features
        )[0]

        lr_classes = list(
            lr_model.classes_
        )

        if lr_pred not in lr_classes:
            raise ValueError("Invalid Logistic Regression class.")

        if 1 not in lr_classes:
            raise ValueError(
                "Logistic Regression model does not contain class 1."
            )

        positive_index = lr_classes.index(1)

        positive_probability = float(
            lr_proba[positive_index]
        )

        lr_confidence = float(
            max(lr_proba)
        )

        severity_result = (
            "High/Critical"
            if lr_pred == 1
            else "Low/Medium"
        )

        risk_level = (
            "HIGH RISK"
            if lr_pred == 1
            else "LOW RISK"
        )

        categories = [
            "Cyber Crime",
            "Drug Offences",
            "Property Crime",
            "Public Order",
            "Violent Crime"
        ]

        dt_pred = int(
            dt_model.predict(features)[0]
        )

        category = (
            categories[dt_pred]
            if 0 <= dt_pred < len(categories)
            else "Unknown"
        )

        result = {

            "crime_type": crime_type,

            "crime_type_conf": (
                f"{rf_confidence * 100:.1f}%"
            ),

            "severity": severity_result,

            "severity_conf": (
                f"{lr_confidence * 100:.1f}%"
            ),

            "positive_probability": (
                f"{positive_probability * 100:.1f}%"
            ),

            "category": category,

            "risk_level": risk_level,

            "models_used": [
                "Random Forest",
                "Logistic Regression",
                "Decision Tree"
            ]

        }

        return jsonify(result)

    except ValueError as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": "Unable to generate crime prediction."
        }), 500


@app.route("/api/health")
def api_health():

    database_status = False

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("SELECT 1")

        cursor.fetchone()

        cursor.close()

        database_status = True

    except Exception as e:

        print("Health check error:", e)

    finally:

        if conn and conn.is_connected():
            conn.close()

    return jsonify({

        "application": "CPAPS",

        "status": "Running",

        "database": (
            "Connected"
            if database_status
            else "Disconnected"
        ),

        "models": (
            "Loaded"
            if models_available()
            else "Unavailable"
        )

    })

@app.errorhandler(404)
def page_not_found(error):

    if request.path.startswith("/api/"):

        return jsonify({
            "error": "API endpoint not found."
        }), 404

    return "Page not found", 404


if __name__ == "__main__":

    print("\n========================================")

    print("CPAPS WEB APPLICATION")

    print("Crime Pattern Analysis and Prediction System")

    print("========================================")

    print("\nApplication components:")

    print(
        "Database:",
        DB_CONFIG["database"]
    )

    print(
        "Machine Learning Models:",
        "Loaded" if models_available() else "Check ml_outputs folder"
    )

    print("\nAvailable pages:")

    print("Dashboard: http://127.0.0.1:5000")

    print("Records: http://127.0.0.1:5000/records")

    print("Prediction: http://127.0.0.1:5000/predict")

    print("Hotspots: http://127.0.0.1:5000/hotspots")

    print("Crime Map: http://127.0.0.1:5000/map")

    print("\nStarting Flask server...")

    print("Press CTRL+C to stop.")

    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
