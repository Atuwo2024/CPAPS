from flask import Flask, render_template, jsonify, request
import mysql.connector
import pandas as pd
import joblib
import numpy as np
import os
from mysql.connector import Error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_outputs")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["JSON_SORT_KEYS"] = False

# Keep your existing database settings here.
# For security, consider moving the password to an environment variable.
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get("CPAPS_DB_PASSWORD", "CHANGE_ME"),
    "database": "cpaps_db"
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def query_db(sql, params=None):
    conn = None
    try:
        conn = get_connection()
        return pd.read_sql_query(sql, conn, params=params)
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
    clean_df = df.astype(object).where(pd.notna(df), None)
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


# Load the new six-input prediction bundle.
prediction_bundle = None
prediction_models = {}
prediction_features = []
prediction_targets = []

bundle_path = os.path.join(MODEL_DIR, "prediction_model_bundle.pkl")

try:
    if os.path.exists(bundle_path):
        prediction_bundle = joblib.load(bundle_path)
        prediction_models = prediction_bundle.get("models", {})
        prediction_features = prediction_bundle.get("features", [])
        prediction_targets = prediction_bundle.get("targets", [])
        print("Successfully loaded: prediction_model_bundle.pkl")
        print("Prediction features:", prediction_features)
        print("Prediction targets:", prediction_targets)
    else:
        print("WARNING: prediction_model_bundle.pkl was not found.")
except Exception as e:
    print("ERROR loading prediction bundle:", e)


REQUIRED_FEATURES = [
    "county",
    "area_type",
    "day_of_week",
    "hour_of_day",