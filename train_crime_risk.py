
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_outputs")

DATA_FILE = os.path.join(
    OUTPUT_DIR, "monthly_crime_patterns.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR, "crime_risk_model.pkl"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR, "crime_risk_classification_report.txt"
)

MATRIX_FILE = os.path.join(
    OUTPUT_DIR, "crime_risk_confusion_matrix.png"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR, "crime_risk_test_predictions.csv"
)

CHART_FILE = os.path.join(
    OUTPUT_DIR, "crime_risk_distribution.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==========================================")
print(" CPAPS - CRIME RISK PREDICTION")
print("==========================================")


if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

required_columns = [
    "county",
    "sub_county",
    "ward",
    "month",
    "total_cases"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("\nDataset loaded successfully.")
print("Total records:", len(df))


data = df[required_columns].copy()

location_columns = [
    "county",
    "sub_county",
    "ward"
]

for col in location_columns:
    data[col] = (
        data[col]
        .astype("string")
        .str.strip()
    )

    data[col] = data[col].replace("", pd.NA)

data["month"] = pd.to_datetime(
    data["month"],
    format="%Y-%m",
    errors="coerce"
)

data["total_cases"] = pd.to_numeric(
    data["total_cases"],
    errors="coerce"
)

data = data.dropna(
    subset=required_columns
).copy()

if (data["total_cases"] < 0).any():
    raise ValueError(
        "Negative crime counts were found."
    )

data = data.sort_values(
    location_columns + ["month"]
).reset_index(drop=True)

if data.duplicated(
    subset=location_columns + ["month"]
).any():
    raise ValueError(
        "Duplicate location-month records found. "
        "Please check monthly_crime_patterns.csv."
    )

print("Records after cleaning:", len(data))

print(
    "Historical period:",
    data["month"].min().strftime("%B %Y"),
    "to",
    data["month"].max().strftime("%B %Y")
)

print(
    "Locations:",
    data[location_columns].drop_duplicates().shape[0]
)

# Each location uses its own previous crime counts.
# Current month's count is NOT used as an input.

grouped = data.groupby(
    location_columns,
    sort=False
)["total_cases"]

for lag in [1, 2, 3, 12]:
    data[f"lag_{lag}"] = grouped.shift(lag)

data["rolling_mean_3"] = (
    data.groupby(location_columns)["total_cases"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

data["month_number"] = data["month"].dt.month

data = data.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3"
    ]
).copy()

if data.empty:
    raise ValueError(
        "Not enough historical months to create "
        "the required lag features."
    )

print("\nRecords available after feature creation:", len(data))


# Use the latest 12 months for testing.
# Earlier months are used for training.

unique_months = sorted(
    data["month"].unique()
)

if len(unique_months) <= 12:
    raise ValueError(
        "Not enough months for a 12-month test period."
    )

test_months = unique_months[-12:]

train_data = data[
    ~data["month"].isin(test_months)
].copy()

test_data = data[
    data["month"].isin(test_months)
].copy()

print("\nTraining period:")
print(
    train_data["month"].min().strftime("%B %Y"),
    "to",
    train_data["month"].max().strftime("%B %Y")
)

print("\nTesting period:")
print(
    test_data["month"].min().strftime("%B %Y"),
    "to",
    test_data["month"].max().strftime("%B %Y")
)

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))


# Risk thresholds are calculated ONLY from
# training-period crime counts.

lower_threshold = train_data["total_cases"].quantile(0.33)
upper_threshold = train_data["total_cases"].quantile(0.67)

def assign_risk(counts):
    return np.select(
        [
            counts <= lower_threshold,
            counts <= upper_threshold
        ],
        [
            "Low",
            "Medium"
        ],
        default="High"
    )

train_data["risk_level"] = assign_risk(
    train_data["total_cases"]
)

test_data["risk_level"] = assign_risk(
    test_data["total_cases"]
)

print("\nRisk thresholds:")
print("Low / Medium boundary:", lower_threshold)
print("Medium / High boundary:", upper_threshold)

print("\nTraining risk distribution:")
print(train_data["risk_level"].value_counts())

print("\nTesting risk distribution:")
print(test_data["risk_level"].value_counts())

if train_data["risk_level"].nunique() < 2:
    raise ValueError(
        "Only one risk class exists in the training data. "
        "The classifier cannot learn separate risk levels. "
        "Review the monthly crime counts and risk thresholds."
    )


categorical_features = [
    "county",
    "sub_county",
    "ward"
]

numerical_features = [
    "month_number",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3"
]

feature_columns = (
    categorical_features + numerical_features
)

X_train = train_data[feature_columns]
y_train = train_data["risk_level"]

X_test = test_data[feature_columns]
y_test = test_data["risk_level"]


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

print("\nTraining crime risk prediction model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print("\n==========================================")
print(" CRIME RISK MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("CPAPS CRIME RISK CLASSIFICATION REPORT\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(
        f"Low/Medium threshold: {lower_threshold}\n"
    )
    file.write(
        f"Medium/High threshold: {upper_threshold}\n\n"
    )
    file.write(report)


labels = list(
    model.named_steps["classifier"].classes_
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("CPAPS - Crime Risk Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MATRIX_FILE,
    dpi=300
)

plt.close()


test_results = test_data[
    location_columns + ["month", "total_cases"]
].copy()

test_results["actual_risk"] = y_test.values
test_results["predicted_risk"] = predictions

test_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

risk_counts = train_data["risk_level"].value_counts()

risk_order = [
    level for level in ["Low", "Medium", "High"]
    if level in risk_counts.index
]

plt.figure(figsize=(8, 5))

plt.bar(
    risk_order,
    [risk_counts[level] for level in risk_order]
)

plt.title("CPAPS - Historical Crime Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Location-Month Records")

plt.tight_layout()

plt.savefig(
    CHART_FILE,
    dpi=300
)

plt.close()


model_bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "risk_thresholds": {
        "low_medium": float(lower_threshold),
        "medium_high": float(upper_threshold)
    },
    "risk_classes": labels,
    "accuracy": float(accuracy),
    "training_period": [
        str(train_data["month"].min().date()),
        str(train_data["month"].max().date())
    ],
    "testing_period": [
        str(test_data["month"].min().date()),
        str(test_data["month"].max().date())
    ]
}

joblib.dump(
    model_bundle,
    MODEL_FILE
)

print("\n==========================================")
print(" CRIME RISK MODEL COMPLETED")
print("==========================================")

print("\nFiles generated:")

print("1.", MODEL_FILE)
print("2.", REPORT_FILE)
print("3.", MATRIX_FILE)
print("4.", PREDICTIONS_FILE)
print("5.", CHART_FILE)

print("\nCPAPS crime risk model saved successfully.")
