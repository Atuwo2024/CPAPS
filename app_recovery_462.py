from flask import Flask, render_template, jsonify, request
import os
import math
import traceback

import joblib
import mysql.connector
import numpy as np
import pandas as pd
from mysql.connector import Error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_outputs")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)
app.config["JSON_SORT_KEYS"] = False

# Set this in PowerShell before starting Flask:
# $env:CPAPS_DB_PASSWORD="your_mysql_password"
DB_CONFIG = {
    "host": os.environ.get("CPAPS_DB_HOST", "localhost"),
    "user": os.environ.get("CPAPS_DB_USER", "root"),
    "password": os.environ.get("12345"),
    "database": os.environ.get("CPAPS_DB_NAME", "cpaps_db"),
}

# The feature order must match the order used to train the pipelines.
REQUIRED_FEATURES = [
    "county",
    "area_type",
    "day_of_week",
    "hour_of_day",
    "month_number",
    "year_number",
]

# Model bundle target names. The database stores injuries_reported as a
# boolean (0/1), not as a count of injured people.
REQUIRED_TARGETS = [
    "crime_type",
    "crime_category",
    "severity_level",
    "property_loss_ksh",
    "injuries_reported",
    "fatalities",
]

prediction_bundle = None
prediction_models = {}
prediction_features = []
prediction_targets = []
model_load_error = None

bundle_path = os.path.join(MODEL_DIR, "prediction_model_bundle.pkl")
try:
    if os.path.exists(bundle_path):
        prediction_bundle = joblib.load(bundle_path)
        prediction_models = prediction_bundle.get("models", {})
        prediction_features = prediction_bundle.get("features", [])
        prediction_targets = prediction_bundle.get("targets", [])
        print("Loaded prediction_model_bundle.pkl")
        print("Prediction features:", prediction_features)
        print("Prediction targets:", prediction_targets)
    else:
        model_load_error = "prediction_model_bundle.pkl was not found."
        print("WARNING:", model_load_error)
except Exception as exc:
    model_load_error = str(exc)
    print("ERROR loading prediction bundle:", exc)


def get_connection():
    """Create a fresh MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


def query_db(sql, params=None):
    """Run a SELECT query and return its result as a DataFrame."""
    conn = None
    try:
        conn = get_connection()
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


def dataframe_to_json(df):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if df is None or df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def json_error(message, status=500):
    return jsonify({"success": False, "error": str(message)}), status


def get_report():
    """Read the reporting view used by the dashboard and records pages."""
    return query_db("SELECT * FROM v_crime_report")


def model_status():
    required = {
        "crime_type",
        "crime_category",
        "severity_level",
        "property_loss_ksh",
        "injuries_reported",
        "fatalities",
    }
    return (
        prediction_bundle is not None
        and isinstance(prediction_models, dict)
        and required.issubset(set(prediction_models.keys()))
        and prediction_features == REQUIRED_FEATURES
    )


def get_numeric(value, field, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a valid finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must not exceed {maximum}.")
    return int(number) if number.is_integer() else number


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/records")
def records():
    return render_template("records.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/hotspots")
def hotspots():
    return render_template("hotspots.html")


@app.route("/map")
def crime_map():
    return render_template("map.html")


@app.route("/api/stats")
def api_stats():
    try:
        df = get_report()
        if df.empty:
            return jsonify({
                "total_incidents": 0,
                "open_cases": 0,
                "total_loss": 0,
                "fatalities": 0,
                "top_county": "N/A",
                "top_crime": "N/A",
            })

        status = df["status"].astype(str).str.lower() if "status" in df else pd.Series([], dtype=str)
        return jsonify({
            "total_incidents": int(len(df)),
            "open_cases": int(status.isin(["open", "under investigation"]).sum()) if len(status) else 0,
            "total_loss": float(pd.to_numeric(df.get("property_loss_ksh", 0), errors="coerce").fillna(0).sum()),
            "fatalities": int(pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0).sum()),
            "top_county": str(df["county"].mode().iloc[0]) if "county" in df and not df["county"].dropna().empty else "N/A",
            "top_crime": str(df["crime_type"].mode().iloc[0]) if "crime_type" in df and not df["crime_type"].dropna().empty else "N/A",
        })
    except Exception as exc:
        return json_error(exc)


@app.route("/api/crime_by_type")
def api_crime_by_type():
    try:
        df = query_db("""
            SELECT crime_type, COUNT(*) AS total
            FROM v_crime_report
            GROUP BY crime_type
            ORDER BY total DESC
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/crime_by_county")
def api_crime_by_county():
    try:
        df = query_db("""
            SELECT county, COUNT(*) AS total
            FROM v_crime_report
            GROUP BY county
            ORDER BY total DESC
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/crime_by_hour")
def api_crime_by_hour():
    try:
        df = query_db("""
            SELECT HOUR(incident_time) AS hour_of_day, COUNT(*) AS total
            FROM crime_incidents
            GROUP BY HOUR(incident_time)
            ORDER BY hour_of_day
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/monthly_trend")
def api_monthly_trend():
    try:
        df = query_db("""
            SELECT year_number, month_number, COUNT(*) AS total
            FROM v_crime_report
            GROUP BY year_number, month_number
            ORDER BY year_number, month_number
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/severity")
def api_severity():
    try:
        df = query_db("""
            SELECT severity_level, COUNT(*) AS total
            FROM v_crime_report
            GROUP BY severity_level
            ORDER BY total DESC
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/crime_by_day")
def api_crime_by_day():
    try:
        df = query_db("""
            SELECT day_of_week, COUNT(*) AS total
            FROM v_crime_report
            GROUP BY day_of_week
            ORDER BY FIELD(day_of_week,
                'Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday')
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)

@app.route("/api/records")
def api_records():
    try:
        limit = request.args.get("limit", default=1000, type=int)
        limit = max(1, min(limit, 5000))
        df = query_db(
            "SELECT * FROM v_crime_report ORDER BY incident_date DESC, incident_time DESC LIMIT %s",
            (limit,),
        )
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)

@app.route("/api/hotspots")
def api_hotspots():
    try:
        # Derive hotspot summaries from the reporting view. This avoids
        # depending on a separate hotspot table being pre-populated.
        df = query_db("""
            SELECT
                county,
                sub_county,
                ward,
                area_type,
                COUNT(*) AS total_crimes,
                AVG(latitude) AS latitude,
                AVG(longitude) AS longitude
            FROM v_crime_report
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            GROUP BY county, sub_county, ward, area_type
            ORDER BY total_crimes DESC
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)


@app.route("/api/map_data")
def api_map_data():
    try:
        df = query_db("""
            SELECT
                case_number, crime_type, crime_category, severity_level,
                county, sub_county, ward, area_type,
                latitude, longitude, incident_date, incident_time
            FROM v_crime_report
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY incident_date DESC
        """)
        return jsonify(dataframe_to_json(df))
    except Exception as exc:
        return json_error(exc)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model_status():
        return json_error(
            "The six-model prediction bundle is not ready. Check ml_outputs/prediction_model_bundle.pkl and its feature order.",
            503,
        )

    data = request.get_json(silent=True) or {}

    required_input_fields = REQUIRED_FEATURES
    missing = [field for field in required_input_fields if field not in data or data[field] in (None, "")]
    if missing:
        return json_error("Missing input fields: " + ", ".join(missing), 400)

    county = str(data["county"]).strip()
    area_type = str(data["area_type"]).strip()
    day_of_week = str(data["day_of_week"]).strip()

    if not county or not area_type or not day_of_week:
        return json_error("County, area type and day of week cannot be blank.", 400)

    try:
        hour = get_numeric(data["hour_of_day"], "hour_of_day", 0, 23)
        month = get_numeric(data["month_number"], "month_number", 1, 12)
        year = get_numeric(data["year_number"], "year_number", 1900, 2100)
    except ValueError as exc:
        return json_error(exc, 400)

    input_df = pd.DataFrame([{
        "county": county,
        "area_type": area_type,
        "day_of_week": day_of_week,
        "hour_of_day": int(hour),
        "month_number": int(month),
        "year_number": int(year),
    }], columns=REQUIRED_FEATURES)

    try:
        results = {}
        for target in REQUIRED_TARGETS:
            model = prediction_models[target]
            predicted = model.predict(input_df)[0]

            if isinstance(predicted, np.generic):
                predicted = predicted.item()

            if target == "property_loss_ksh":
                predicted = max(0.0, float(predicted))
                predicted = round(predicted, 2)
            elif target == "fatalities":
                predicted = max(0, int(round(float(predicted))))
            elif target == "injuries_reported":
                predicted = bool(int(round(float(predicted))))
            else:
                predicted = str(predicted)

            results[target] = predicted

        severity = str(results["severity_level"]).strip().lower()
        if severity == "critical":
            risk_level = "Critical"
        elif severity == "high":
            risk_level = "High"
        elif severity == "medium":
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "success": True,
            "crime_type": results["crime_type"],
            "crime_category": results["crime_category"],
            "category": results["crime_category"],
            "severity_level": results["severity_level"],
            "severity": results["severity_level"],
            "property_loss_ksh": results["property_loss_ksh"],
            "injuries_reported": results["injuries_reported"],
            "fatalities": results["fatalities"],
            "risk_level": risk_level,
            "experimental": True,
            "warning": (
                "Experimental estimates only. Current evaluation results are weak; "
                "these predictions must not be used as confirmed outcomes or for operational decisions."
            ),
            "models_used": REQUIRED_TARGETS,
        })
    except Exception as exc:
        print("Prediction error:", traceback.format_exc())
        return json_error("Prediction failed: " + str(exc), 500)

@app.route("/api/health")
def api_health():
    database_status = "Disconnected"
    database_error = None
    try:
        conn = get_connection()
        conn.close()
        database_status = "Connected"
    except Exception as exc:
        database_error = str(exc)

    models = "Loaded" if model_status() else "Unavailable"

    response = {
        "application": "CPAPS",
        "database": database_status,
        "models": models,
        "status": "Running",
    }
    if database_error:
        response["database_error"] = database_error
    if model_load_error:
        response["model_error"] = model_load_error

    return jsonify(response)

@app.errorhandler(404)
def page_not_found(error):
    if request.path.startswith("/api/"):
        return json_error("API endpoint not found.", 404)
    return "Page not found.", 404


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith("/api/"):
        return json_error("An internal server error occurred.", 500)
    return "An internal server error occurred.", 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
