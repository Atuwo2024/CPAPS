import csv
import io
import pandas as pd
import os
import math
import re
import secrets
import traceback
from datetime import datetime

from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
import joblib
import mysql.connector
import numpy as np
import pandas as pd
from mysql.connector import Error, IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_outputs")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = os.environ.get("CPAPS_SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("CPAPS_SECRET_KEY"):
    app.logger.warning("Set CPAPS_SECRET_KEY to keep sessions valid across restarts.")
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("CPAPS_COOKIE_SECURE", "").lower() == "true",
)

# Set this in PowerShell before starting Flask:
# $env:CPAPS_DB_PASSWORD="your_mysql_password"
DB_CONFIG = {
    "host": os.environ.get("CPAPS_DB_HOST", "localhost"),
    "user": os.environ.get("CPAPS_DB_USER", "root"),
    "password": os.environ.get("CPAPS_DB_PASSWORD", ""),
    "database": os.environ.get("CPAPS_DB_NAME", "cpaps_db"),
}
app.config["USERS_TABLE_READY"] = False

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

        prediction_features = prediction_bundle.get("features", [])
        prediction_targets = prediction_bundle.get("targets", [])

        prediction_models = {}

        saved_models = prediction_bundle.get("models", {})

        for target, model_file in saved_models.items():
            model_path = os.path.join(MODEL_DIR, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_file}"
                )

            prediction_models[target] = joblib.load(model_path)

            if not callable(getattr(prediction_models[target], "predict", None)):
                raise TypeError(
                    f"{model_file} is not a valid prediction model."
                )

        print("Loaded prediction_model_bundle.pkl")
        print("Loaded individual trained models:", list(prediction_models.keys()))
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


def authentication_database_error(error):
    error_number = getattr(error, "errno", None)
    if error_number == 1045:
        return (
            "MySQL rejected the configured account. Set the correct "
            "CPAPS_DB_USER and CPAPS_DB_PASSWORD, then restart the app."
        )
    if error_number == 1049:
        return (
            f"MySQL database '{DB_CONFIG['database']}' does not exist. "
            "Create it by running the project's database schema SQL file."
        )
    if error_number == 1044:
        return (
            f"The configured MySQL account cannot access '{DB_CONFIG['database']}'. "
            "Grant that account access to the database."
        )
    if error_number in {1142, 1143}:
        return (
            "The configured MySQL account lacks permission to create or access "
            "application accounts. Grant it the required table permissions."
        )
    if error_number in {2002, 2003}:
        return (
            f"Cannot reach MySQL at '{DB_CONFIG['host']}'. Start MySQL or check "
            "CPAPS_DB_HOST."
        )
    return (
        "CPAPS cannot initialize authentication in MySQL. Verify that MySQL "
        f"is running and the configured account can access '{DB_CONFIG['database']}'."
    )


def ensure_users_table():
    if app.config["USERS_TABLE_READY"]:
        return

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(64) NOT NULL UNIQUE,
                email VARCHAR(255) NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'viewer',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        app.config["USERS_TABLE_READY"] = True
    finally:
        cursor.close()
        conn.close()


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


@app.context_processor
def inject_template_context():
    return {"csrf_token": csrf_token}


@app.before_request
def authenticate_request():
    csrf_token()
    if request.endpoint in {"static", "login", "initial_setup", "api_health"}:
        return None

    is_api = request.path.startswith("/api/")
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            return (jsonify({"success": False, "error": "Invalid CSRF token."}), 400) if is_api else (
                "Invalid CSRF token.", 400
            )

    try:
        ensure_users_table()
    except Error as exc:
        app.logger.exception("Unable to initialize user authentication.")
        message = authentication_database_error(exc)
        return (jsonify({"success": False, "error": message}), 503) if is_api else (
            message, 503
        )

    user_id = session.get("user_id")
    if not user_id:
        return (jsonify({"success": False, "error": "Authentication required."}), 401) if is_api else (
            redirect(url_for("login")), 302
        )

    try:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT user_id, username, email, role, is_active
                FROM app_users
                WHERE user_id = %s
            """, (user_id,))
            user = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()
    except Error as exc:
        app.logger.exception("Unable to validate the application account.")
        message = authentication_database_error(exc)
        return (jsonify({"success": False, "error": message}), 503) if is_api else (
            message, 503
        )

    if not user or not user[4]:
        session.clear()
        return (jsonify({"success": False, "error": "Authentication required."}), 401) if is_api else (
            redirect(url_for("login")), 302
        )

    g.current_user = {
        "user_id": user[0],
        "username": user[1],
        "email": user[2],
        "role": user[3],
    }

    admin_only = request.path == "/users" or request.path.startswith("/api/users")
    analyst_or_admin = request.path == "/upload" or request.path.startswith("/api/upload")
    if admin_only and user[3] != "admin":
        return (jsonify({"success": False, "error": "Administrator access required."}), 403) if is_api else (
            "Administrator access required.", 403
        )
    if analyst_or_admin and user[3] == "viewer":
        return (jsonify({"success": False, "error": "Upload access required."}), 403) if is_api else (
            "Upload access required.", 403
        )

    return None


def valid_username(value):
    return bool(value and value.strip() and len(value.strip()) <= 64)

def get_app_user_count():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM app_users")
        return int(cursor.fetchone()[0])
    finally:
        cursor.close()
        conn.close()


@app.route("/setup", methods=["GET", "POST"])
def initial_setup():
    try:
        ensure_users_table()
        if get_app_user_count():
            return redirect(url_for("login"))
    except Error as exc:
        app.logger.exception("Unable to initialize the first administrator.")
        return authentication_database_error(exc), 503

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not valid_username(username):
            error = "Enter a username containing 1-64 characters."
        elif len(password) < 8:
            error = "The password must contain at least 8 characters."
        else:
            conn = get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO app_users (username, password_hash, role)
                    VALUES (%s, %s, 'admin')
                """, (username, generate_password_hash(password)))
                conn.commit()
                session.clear()
                csrf_token()
                session["user_id"] = cursor.lastrowid
                return redirect(url_for("dashboard"))
            except IntegrityError:
                conn.rollback()
                error = "That username already exists. Please choose another."
            except Error:
                conn.rollback()
                app.logger.exception("Unable to create the initial administrator.")
                error = "Unable to create the administrator because of a database error."
            finally:
                cursor.close()
                conn.close()

    return render_template("setup.html", error=error)



@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        ensure_users_table()
        if get_app_user_count() == 0:
            return redirect(url_for("initial_setup"))
    except Error as exc:
        app.logger.exception("Unable to check application accounts.")
        return authentication_database_error(exc), 503

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT user_id, password_hash, is_active
                FROM app_users
                WHERE username = %s
            """, (username,))
            user = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        if user and user[2] and check_password_hash(user[1], password):
            session.clear()
            csrf_token()
            session["user_id"] = user[0]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password, or this account is inactive."

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


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

    def convert_value(value):
        if value is None:
            return None

        if isinstance(value, pd.Timedelta):
            return str(value)

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).isoformat()

        if isinstance(value, np.timedelta64):
            return str(pd.Timedelta(value))

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not np.isfinite(value):
            return None

        return value

    records = clean.to_dict(orient="records")

    return [
        {
            key: convert_value(value)
            for key, value in record.items()
        }
        for record in records
    ]

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
        return render_template("crime_map.html")


@app.route("/upload")
def upload_dataset():
    officers = query_db("""
        SELECT officer_id, badge_number, first_name, last_name
        FROM officers
        ORDER BY last_name, first_name
    """)
    return render_template("upload.html", officers=dataframe_to_json(officers))


@app.route("/users")
def users():
    return render_template("users.html")


@app.route("/reports")
def reports():
    return render_template("reports.html")


def parse_upload_row(row, row_number):
    def required(name):
        value = (row.get(name) or "").strip()
        if not value:
            raise ValueError(f"Row {row_number}: {name} is required.")
        return value

    def optional(name, default=""):
        return (row.get(name) or "").strip() or default

    def enum_value(name, allowed, default):
        value = optional(name, default)
        values = {item.lower(): item for item in allowed}
        if value.lower() not in values:
            raise ValueError(
                f"Row {row_number}: {name} must be one of {', '.join(allowed)}."
            )
        return values[value.lower()]

    case_number = required("case_number")
    if len(case_number) > 50:
        raise ValueError(f"Row {row_number}: case_number must be 50 characters or fewer.")
    for field in ("crime_type", "crime_category", "county"):
        if len(required(field)) > 100:
            raise ValueError(f"Row {row_number}: {field} must be 100 characters or fewer.")
    for field in ("sub_county", "ward"):
        if len(optional(field)) > 100:
            raise ValueError(f"Row {row_number}: {field} must be 100 characters or fewer.")

    try:
        incident_date = datetime.strptime(required("incident_date"), "%Y-%m-%d").date()
        reported_date = datetime.strptime(
            optional("reported_date", incident_date.isoformat()), "%Y-%m-%d"
        ).date()
    except ValueError:
        raise ValueError(f"Row {row_number}: dates must use YYYY-MM-DD format.")

    time_value = optional("incident_time", "00:00:00")
    time_value = re.sub(r"^0\s+days?\s+", "", time_value, flags=re.IGNORECASE)
    try:
        incident_time = datetime.strptime(time_value, "%H:%M:%S").time()
    except ValueError:
        try:
            incident_time = datetime.strptime(time_value, "%H:%M").time()
        except ValueError:
            raise ValueError(f"Row {row_number}: incident_time must use HH:MM or HH:MM:SS.")

    try:
        property_loss = float(optional("property_loss_ksh", "0"))
        fatalities = int(optional("fatalities", "0"))
    except ValueError:
        raise ValueError(f"Row {row_number}: property_loss_ksh and fatalities must be numeric.")
    if not math.isfinite(property_loss) or property_loss < 0 or fatalities < 0:
        raise ValueError(f"Row {row_number}: loss and fatalities cannot be negative or invalid.")

    injuries_value = optional("injuries_reported", "0").lower()
    if injuries_value not in {"0", "1", "true", "false", "yes", "no"}:
        raise ValueError(f"Row {row_number}: injuries_reported must be true/false or 1/0.")
    injuries_reported = injuries_value in {"1", "true", "yes"}

    latitude = optional("latitude")
    longitude = optional("longitude")
    if bool(latitude) != bool(longitude):
        raise ValueError(f"Row {row_number}: latitude and longitude must both be provided or omitted.")
    try:
        latitude = float(latitude) if latitude else None
        longitude = float(longitude) if longitude else None
    except ValueError:
        raise ValueError(f"Row {row_number}: coordinates must be numeric.")
    if latitude is not None and (
        not math.isfinite(latitude) or not math.isfinite(longitude)
        or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
    ):
        raise ValueError(f"Row {row_number}: coordinates are outside valid ranges.")

    return {
        "case_number": case_number,
        "incident_date": incident_date,
        "incident_time": incident_time,
        "reported_date": reported_date,
        "crime_type": required("crime_type"),
        "crime_category": required("crime_category"),
        "severity_level": enum_value(
            "severity_level", ["Low", "Medium", "High", "Critical"], ""
        ),
        "county": required("county"),
        "sub_county": optional("sub_county") or None,
        "ward": optional("ward") or None,
        "area_type": enum_value("area_type", ["Urban", "Suburban", "Rural"], "Urban"),
        "latitude": latitude,
        "longitude": longitude,
        "status": enum_value(
            "status", ["Open", "Under Investigation", "Closed", "Cold Case"], "Open"
        ),
        "weather_condition": enum_value(
            "weather_condition", ["Clear", "Rainy", "Cloudy", "Night", "Unknown"], "Unknown"
        ),
        "property_loss_ksh": property_loss,
        "injuries_reported": injuries_reported,
        "fatalities": fatalities,
        "description": optional("description") or None,
    }


@app.route("/api/upload", methods=["POST"])
def api_upload_dataset():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return json_error("Choose a CSV or Excel file to upload.", 400)

    filename = uploaded_file.filename.lower()
    if not filename.endswith((".csv", ".xlsx")):
        return json_error("Only CSV and Excel (.xlsx) files are supported.", 400)
    try:
        officer_id = int(request.form.get("officer_id", ""))
    except ValueError:
        return json_error("Choose a valid reporting officer.", 400)

    
    required_headers = {
        "case_number", "incident_date", "crime_type", "crime_category",
        "severity_level", "county",
    }

    try:
        file_bytes = uploaded_file.read()

        if filename.endswith(".csv"):
            contents = file_bytes.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(contents))

            if not reader.fieldnames:
                return json_error("The CSV has no header row.", 400)

            fieldnames = [
                (name or "").strip().lower()
                for name in reader.fieldnames
            ]
            reader.fieldnames = fieldnames
            raw_rows = reader

        else:
            try:
                worksheet = pd.read_excel(
                    io.BytesIO(file_bytes),
                    header=None,
                    dtype=object,
                    engine="openpyxl"
                )
            except Exception:
                return json_error(
                    "The Excel file could not be read. Please check the workbook.",
                    400
                )

            if worksheet.empty:
                return json_error("The Excel file is empty.", 400)

            fieldnames = [
                "" if pd.isna(name) else str(name).strip().lower()
                for name in worksheet.iloc[0].tolist()
            ]

            raw_rows = [
                dict(zip(fieldnames, values))
                for values in worksheet.iloc[1:].itertuples(
                    index=False, name=None
                )
            ]

        if len(set(fieldnames)) != len(fieldnames):
            return json_error("The file contains duplicate column names.", 400)

        missing_headers = sorted(required_headers - set(fieldnames))
        if missing_headers:
            return json_error(
                "Missing required columns: " + ", ".join(missing_headers),
                400
            )

        rows = []
        seen_case_numbers = set()

        for row_number, raw_row in enumerate(raw_rows, start=2):

            if None in raw_row:
                raise ValueError(
                    f"Row {row_number}: more data cells than header columns."
                )

            cleaned_row = {}

            for name, value in raw_row.items():
                if name is None:
                    continue

                if pd.isna(value):
                    value = ""
                elif name in {"incident_date", "reported_date"} and hasattr(value, "strftime"):
                    value = value.strftime("%Y-%m-%d")
                elif name == "incident_time" and hasattr(value, "strftime"):
                    value = value.strftime("%H:%M:%S")
                else:
                    value = str(value).strip()

                cleaned_row[name] = value

            if not any(value for value in cleaned_row.values()):
                continue

            row = parse_upload_row(cleaned_row, row_number)

            if row["case_number"] in seen_case_numbers:
                raise ValueError(
                    f"Row {row_number}: duplicate case_number in this file."
                )

            seen_case_numbers.add(row["case_number"])
            rows.append(row)

            if len(rows) > 5000:
                raise ValueError(
                    "A single upload cannot contain more than 5,000 records."
                )

        if not rows:
            return json_error("The file contains no incident rows.", 400)

    except UnicodeDecodeError:
        return json_error("The CSV must be encoded as UTF-8.", 400)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except csv.Error:
        return json_error("The uploaded file is not a valid CSV.", 400)


    conn = get_connection()
    cursor = conn.cursor()
    try:
        conn.start_transaction()
        cursor.execute("SELECT officer_id FROM officers WHERE officer_id = %s", (officer_id,))
        if not cursor.fetchone():
            conn.rollback()
            return json_error("The selected reporting officer does not exist.", 400)

        for row in rows:
            cursor.execute(
                "SELECT incident_id FROM crime_incidents WHERE case_number = %s",
                (row["case_number"],),
            )
            if cursor.fetchone():
                raise ValueError(
                    f"Case number {row['case_number']} already exists; no rows were imported."
                )

            cursor.execute("""
                SELECT crime_type_id
                FROM crime_types
                WHERE category = %s AND subcategory = %s AND severity_level = %s
                LIMIT 1
            """, (row["crime_category"], row["crime_type"], row["severity_level"]))
            crime_type = cursor.fetchone()
            if crime_type:
                crime_type_id = crime_type[0]
            else:
                cursor.execute("""
                    INSERT INTO crime_types (category, subcategory, severity_level)
                    VALUES (%s, %s, %s)
                """, (row["crime_category"], row["crime_type"], row["severity_level"]))
                crime_type_id = cursor.lastrowid

            cursor.execute("""
                SELECT location_id
                FROM locations
                WHERE county = %s AND sub_county <=> %s AND ward <=> %s
                  AND area_type = %s AND latitude <=> %s AND longitude <=> %s
                LIMIT 1
            """, (
                row["county"], row["sub_county"], row["ward"], row["area_type"],
                row["latitude"], row["longitude"],
            ))
            location = cursor.fetchone()
            if location:
                location_id = location[0]
            else:
                cursor.execute("""
                    INSERT INTO locations (
                        county, sub_county, ward, area_type, latitude, longitude
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    row["county"], row["sub_county"], row["ward"], row["area_type"],
                    row["latitude"], row["longitude"],
                ))
                location_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO crime_incidents (
                    case_number, crime_type_id, location_id, officer_id,
                    incident_date, incident_time, reported_date, description, status,
                    weather_condition, property_loss_ksh, injuries_reported, fatalities
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row["case_number"], crime_type_id, location_id, officer_id,
                row["incident_date"], row["incident_time"], row["reported_date"],
                row["description"], row["status"], row["weather_condition"],
                row["property_loss_ksh"], row["injuries_reported"], row["fatalities"],
            ))

        conn.commit()
        return jsonify({"success": True, "imported": len(rows)})
    except ValueError as exc:
        conn.rollback()
        return json_error(exc, 400)
    except Error:
        conn.rollback()
        app.logger.exception("Dataset upload failed.")
        return json_error("Dataset import failed; no rows were imported.", 500)
    finally:
        cursor.close()
        conn.close()


@app.route("/api/users", methods=["GET", "POST"])
def api_users():
    if request.method == "GET":
        try:
            users_df = query_db("""
                SELECT user_id, username, email, role, is_active, created_at
                FROM app_users
                ORDER BY created_at DESC
            """)
            return jsonify(dataframe_to_json(users_df))
        except Error:
            app.logger.exception("Unable to load application accounts.")
            return json_error("Unable to load application accounts.", 500)

    
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    role = str(data.get("role", "viewer")).strip().lower()

    if not valid_username(username):
        return json_error("Enter a username containing 1-64 characters.", 400)

    if len(password) < 8:
        return json_error("The password must contain at least 8 characters.", 400)

    if role not in {"admin", "analyst", "viewer"}:
        return json_error("Role must be admin, analyst, or viewer.", 400)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO app_users (username, password_hash, role)
            VALUES (%s, %s, %s)
        """, (username, generate_password_hash(password), role))

        conn.commit()
        return jsonify({"success": True, "user_id": cursor.lastrowid}), 201

    except IntegrityError:
        conn.rollback()
        return json_error("That username already exists. Please choose another.", 409)

    except Error:
        conn.rollback()
        app.logger.exception("Unable to create application account.")
        return json_error("Unable to create account because of a database error.", 500)

    finally:
        cursor.close()
        conn.close()



@app.route("/api/users/<int:user_id>", methods=["PATCH"])
def api_update_user(user_id):
    data = request.get_json(silent=True) or {}
    updates = {}
    if "role" in data:
        role = str(data["role"]).strip().lower()
        if role not in {"admin", "analyst", "viewer"}:
            return json_error("Role must be admin, analyst, or viewer.", 400)
        updates["role"] = role
    
    if "is_active" in data:
        if not isinstance(data["is_active"], bool):
            return json_error("is_active must be true or false.", 400)
        updates["is_active"] = data["is_active"]

    if "password" in data:
        password = str(data["password"])
        if len(password) < 8:
            return json_error("The password must contain at least 8 characters.", 400)
        updates["password_hash"] = generate_password_hash(password)

    if not updates:
        return json_error("Provide a role, is_active, or password to update.", 400)
    if user_id == g.current_user["user_id"] and (
        updates.get("role", "admin") != "admin" or updates.get("is_active") is False
    ):
        return json_error("You cannot demote or deactivate your own account.", 400)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        conn.start_transaction()
        cursor.execute("""
            SELECT role, is_active FROM app_users WHERE user_id = %s FOR UPDATE
        """, (user_id,))
        existing = cursor.fetchone()
        if not existing:
            conn.rollback()
            return json_error("Account not found.", 404)

        next_role = updates.get("role", existing[0])
        next_active = updates.get("is_active", bool(existing[1]))
        if existing[0] == "admin" and existing[1] and (
            next_role != "admin" or not next_active
        ):
            cursor.execute("""
                SELECT COUNT(*) FROM app_users
                WHERE role = 'admin' AND is_active = TRUE AND user_id != %s
            """, (user_id,))
            if cursor.fetchone()[0] == 0:
                conn.rollback()
                return json_error("The last active administrator cannot be demoted or deactivated.", 400)

        assignments = ", ".join(f"{column} = %s" for column in updates)
        cursor.execute(
            f"UPDATE app_users SET {assignments} WHERE user_id = %s",
            tuple(updates.values()) + (user_id,),
        )
        conn.commit()
        return jsonify({"success": True})
    except Error:
        conn.rollback()
        app.logger.exception("Unable to update application account.")
        return json_error("Unable to update account.", 500)
    finally:
        cursor.close()
        conn.close()

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
        COALESCE(SUM(fatalities), 0) AS total_fatalities,
        COALESCE(SUM(property_loss_ksh), 0) AS total_loss_ksh,
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


@app.errorhandler(413)
def request_entity_too_large(error):
    if request.path.startswith("/api/"):
        return json_error("The uploaded file exceeds the 10 MB limit.", 413)
    return "The uploaded file exceeds the 10 MB limit.", 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)



