
# ============================================================
# CPAPS - MACHINE LEARNING TRAINING ENGINE
# Crime Pattern Analysis and Prediction System
# ============================================================

import os
import warnings
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from mysql.connector import Error

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "ml_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

sns.set_theme(style="darkgrid")

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "database": "cpaps_db"
}

# Feature definitions

FEATURES = [
    "hour_of_day",
    "month_number",
    "year_number",
    "county_enc",
    "area_type_enc",
    "day_enc",
    "severity_enc",
    "property_loss_ksh",
    "injuries_reported",
    "fatalities"
]

# Severity prediction must not use severity_enc.
SEVERITY_FEATURES = [
    "hour_of_day",
    "month_number",
    "year_number",
    "county_enc",
    "area_type_enc",
    "day_enc",
    "property_loss_ksh",
    "injuries_reported",
    "fatalities"
]

ENCODER_COLUMNS = {
    "county": "county_enc",
    "area_type": "area_type_enc",
    "day_of_week": "day_enc",
    "severity_level": "severity_enc",
    "crime_category": "category_enc",
    "crime_type": "crime_type_enc"
}

# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

def get_connection():

    return mysql.connector.connect(**DB_CONFIG)


# ============================================================
# 3. LOAD DATA FROM MYSQL
# ============================================================

def load_data():

    print("\nLoading crime data from MySQL...")

    conn = None

    try:

        conn = get_connection()

        df = pd.read_sql_query(
            "SELECT * FROM v_crime_report",
            conn
        )

        print(f"Records loaded: {len(df)}")

        if df.empty:
            raise ValueError(
                "No crime records found in v_crime_report."
            )

        return df

    except Error as e:

        print("Database error:", e)
        raise

    finally:

        if conn and conn.is_connected():
            conn.close()


# ============================================================
# 4. DATA CLEANING AND FEATURE ENGINEERING
# ============================================================

def prepare_data(df):

    print("\nPreparing machine-learning dataset...")

    df = df.copy()

    required_columns = [
        "hour_of_day",
        "month_number",
        "year_number",
        "county",
        "area_type",
        "day_of_week",
        "severity_level",
        "property_loss_ksh",
        "injuries_reported",
        "fatalities",
        "crime_category",
        "crime_type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing database columns: {missing_columns}"
        )

    # Numeric cleaning

    numeric_columns = [
        "hour_of_day",
        "month_number",
        "year_number",
        "property_loss_ksh",
        "injuries_reported",
        "fatalities"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    # Convert numeric values

    df["hour_of_day"] = df["hour_of_day"].astype(int)

    df["month_number"] = df["month_number"].astype(int)

    df["year_number"] = df["year_number"].astype(int)

    df["property_loss_ksh"] = (
        df["property_loss_ksh"].astype(float)
    )

    df["injuries_reported"] = (
        df["injuries_reported"].astype(int)
    )

    df["fatalities"] = df["fatalities"].astype(int)

    # Validate numerical ranges

    if not df["hour_of_day"].between(0, 23).all():
        raise ValueError("Invalid hour_of_day values.")

    if not df["month_number"].between(1, 12).all():
        raise ValueError("Invalid month_number values.")

    if not df["year_number"].between(1900, 2100).all():
        raise ValueError("Invalid year_number values.")

    if (
        df["property_loss_ksh"].lt(0).any()
        or df["injuries_reported"].lt(0).any()
        or df["fatalities"].lt(0).any()
    ):
        raise ValueError("Negative financial loss or casualty values found.")

    # Separate label encoders for every categorical feature

    encoders = {}

    for column, encoded_column in ENCODER_COLUMNS.items():

        encoder = LabelEncoder()

        df[column] = df[column].astype(str).str.strip()

        df[encoded_column] = encoder.fit_transform(
            df[column]
        )

        encoders[column] = encoder

    print("Data preparation completed.")

    print("Records available:", len(df))

    print("\nEncoded categories:")

    for column, encoder in encoders.items():

        print(f"{column}: {list(encoder.classes_)}")

    return df, encoders


# ============================================================
# 5. DATASET VALIDATION
# ============================================================

def validate_training_data(df):

    if len(df) < 10:

        raise ValueError(
            "At least 10 valid records are required for this training script."
        )

    targets = [
        "crime_type_enc",
        "severity_level",
        "category_enc"
    ]

    for target in targets:

        if df[target].nunique() < 2:

            raise ValueError(
                f"{target} requires at least two different classes."
            )

    print("\nDataset validation completed.")

    print("Crime types:", df["crime_type"].nunique())

    print("Crime categories:", df["crime_category"].nunique())

    print("Severity levels:", df["severity_level"].nunique())

    if len(df) < 100:

        print(
            "\nWARNING: Your dataset is small. "
            "Model performance may not generalise reliably."
        )


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\n{model_name} Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return accuracy, predictions


def cross_validate_model(model, X, y, model_name):

    class_counts = y.value_counts()

    minimum_class_count = int(class_counts.min())

    if minimum_class_count < 2:

        print(
            f"{model_name}: Cross-validation skipped. "
            "At least two records per class are required."
        )

        return None

    folds = min(5, minimum_class_count)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=folds,
        scoring="accuracy"
    )

    print(
        f"{model_name} {folds}-Fold CV: "
        f"{scores.mean() * 100:.2f}% "
        f"(+/- {scores.std() * 100:.2f}%)"
    )

    return scores


def save_confusion_matrix(
    y_test,
    predictions,
    class_names,
    title,
    filename
):

    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=150
    )

    plt.close()


# ============================================================
# 7. RANDOM FOREST - CRIME TYPE PREDICTION
# ============================================================

def train_random_forest(df):

    print("\n" + "=" * 60)

    print("MODEL 1: RANDOM FOREST")

    print("Crime Type Prediction")

    print("=" * 60)

    X = df[FEATURES]

    y = df["crime_type_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    accuracy, predictions = evaluate_model(
        model,
        X_test,
        y_test,
        "Random Forest"
    )

    cross_validate_model(
        model,
        X,
        y,
        "Random Forest"
    )

    # Feature importance

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(11, 6))

    importances.plot(kind="bar")

    plt.title("Random Forest - Feature Importance")

    plt.ylabel("Importance Score")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "rf_feature_importance.png"
        )
    )

    plt.close()

    crime_classes = [
        str(x) for x in
        joblib.load(
            os.path.join(
                OUTPUT_DIR,
                "temporary_placeholder.pkl"
            )
        )
    ] if False else None

    return model, accuracy, predictions, y_test


# ============================================================
# 8. LOGISTIC REGRESSION - SEVERITY PREDICTION
# ============================================================

def train_logistic_regression(df):

    print("\n" + "=" * 60)

    print("MODEL 2: LOGISTIC REGRESSION")

    print("Crime Severity Prediction")

    print("=" * 60)

    X = df[SEVERITY_FEATURES]

    # Binary target

    y = df["severity_level"].isin(
        ["Critical", "High"]
    ).astype(int)

    if y.nunique() < 2:

        raise ValueError(
            "Logistic Regression requires both severity classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    accuracy, predictions = evaluate_model(
        model,
        X_test_scaled,
        y_test,
        "Logistic Regression"
    )

    # Leakage-safe cross-validation pipeline

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    )

    cross_validate_model(
        pipeline,
        X,
        y,
        "Logistic Regression"
    )

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Low/Medium",
            "High/Critical"
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    disp.plot(
        ax=ax,
        cmap="Oranges",
        colorbar=False
    )

    plt.title(
        "Logistic Regression - Severity Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lr_confusion_matrix.png"
        )
    )

    plt.close()

    return model, scaler, accuracy, predictions, y_test


# ============================================================
# 9. DECISION TREE - CRIME CATEGORY PREDICTION
# ============================================================

def train_decision_tree(df):

    print("\n" + "=" * 60)

    print("MODEL 3: DECISION TREE")

    print("Crime Category Prediction")

    print("=" * 60)

    X = df[FEATURES]

    y = df["category_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy, predictions = evaluate_model(
        model,
        X_test,
        y_test,
        "Decision Tree"
    )

    cross_validate_model(
        model,
        X,
        y,
        "Decision Tree"
    )

    # Decision tree visualisation

    plt.figure(figsize=(24, 12))

    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=[
            str(c) for c in model.classes_
        ],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title(
        "CPAPS - Decision Tree Crime Category Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "decision_tree_visual.png"
        ),
        dpi=150
    )

    plt.close()

    # Feature importance

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(11, 6))

    importances.plot(kind="bar")

    plt.title(
        "Decision Tree - Feature Importance"
    )

    plt.ylabel("Importance Score")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "dt_feature_importance.png"
        )
    )

    plt.close()

    return model, accuracy, predictions, y_test


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

def model_comparison(rf_acc, lr_acc, dt_acc):

    names = [
        "Random Forest",
        "Logistic Regression",
        "Decision Tree"
    ]

    accuracies = [
        rf_acc * 100,
        lr_acc * 100,
        dt_acc * 100
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        names,
        accuracies
    )

    plt.title(
        "CPAPS - Machine Learning Model Comparison"
    )

    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)

    for bar, value in zip(bars, accuracies):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.png"
        )
    )

    plt.close()

    best_index = int(np.argmax(accuracies))

    print("\n" + "=" * 60)

    print("CPAPS MODEL PERFORMANCE SUMMARY")

    print("=" * 60)

    for name, accuracy in zip(names, accuracies):

        print(f"{name}: {accuracy:.2f}%")

    print(
        f"\nBest measured accuracy: {names[best_index]}"
    )

    print("=" * 60)


# ============================================================
# 11. SAVE ALL MODELS AND PREPROCESSING OBJECTS
# ============================================================

def save_models(
    rf_model,
    lr_model,
    lr_scaler,
    dt_model,
    encoders
):

    print("\nSaving trained models...")

    joblib.dump(
        rf_model,
        os.path.join(
            OUTPUT_DIR,
            "random_forest_model.pkl"
        )
    )

    joblib.dump(
        lr_model,
        os.path.join(
            OUTPUT_DIR,
            "logistic_regression_model.pkl"
        )
    )

    joblib.dump(
        lr_scaler,
        os.path.join(
            OUTPUT_DIR,
            "lr_scaler.pkl"
        )
    )

    joblib.dump(
        dt_model,
        os.path.join(
            OUTPUT_DIR,
            "decision_tree_model.pkl"
        )
    )

    joblib.dump(
        encoders,
        os.path.join(
            OUTPUT_DIR,
            "label_encoders.pkl"
        )
    )

    metadata = {

        "features": FEATURES,

        "severity_features": SEVERITY_FEATURES,

        "crime_type_classes": (
            encoders["crime_type"].classes_.tolist()
        ),

        "category_classes": (
            encoders["crime_category"].classes_.tolist()
        ),

        "severity_classes": [
            "Low/Medium",
            "High/Critical"
        ],

        "model_version": "2.0",

        "random_state": RANDOM_STATE

    }

    joblib.dump(
        metadata,
        os.path.join(
            OUTPUT_DIR,
            "model_metadata.pkl"
        )
    )

    print("All models and encoders saved successfully.")

    print("Output directory:", OUTPUT_DIR)


# ============================================================
# 12. MAIN TRAINING PROGRAM
# ============================================================

def main():

    print("\n")

    print("=" * 65)

    print("CRIME PATTERN ANALYSIS AND PREDICTION SYSTEM")

    print("CPAPS MACHINE LEARNING TRAINING ENGINE")

    print("=" * 65)

    try:

        # Load database data

        df = load_data()

        # Clean and encode data

        df, encoders = prepare_data(df)

        # Validate dataset

        validate_training_data(df)

        # Train Random Forest

        rf_model, rf_acc, rf_predictions, rf_test = (
            train_random_forest(df)
        )

        # Train Logistic Regression

        lr_model, lr_scaler, lr_acc, lr_predictions, lr_test = (
            train_logistic_regression(df)
        )

        # Train Decision Tree

        dt_model, dt_acc, dt_predictions, dt_test = (
            train_decision_tree(df)
        )

        # Save models and encoders

        save_models(
            rf_model,
            lr_model,
            lr_scaler,
            dt_model,
            encoders
        )

        # Compare model performance

        model_comparison(
            rf_acc,
            lr_acc,
            dt_acc
        )

        print("\nCPAPS MODEL TRAINING COMPLETED!")

        print("All trained models are available in ml_outputs.")

    except Exception as e:

        print("\nTRAINING ERROR:")

        print(e)

        raise

if __name__ == "__main__":

    main()
