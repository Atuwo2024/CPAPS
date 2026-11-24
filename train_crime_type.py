
import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
    OUTPUT_DIR, "cleaned_crime_data.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR, "crime_type_model.pkl"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR, "crime_type_classification_report.txt"
)

MATRIX_FILE = os.path.join(
    OUTPUT_DIR, "crime_type_confusion_matrix.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n======================================")
print(" CPAPS - CRIME TYPE PREDICTION")
print("======================================")



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

print("\nDataset loaded successfully.")
print("Total records:", len(df))

target = "crime_type"

categorical_features = [
    "county",
    "sub_county",
    "ward",
    "area_type",
    "day_of_week"
]

numerical_features = [
    "month_number",
    "year_number",
    "hour_of_day"
]

required_columns = (
    categorical_features
    + numerical_features
    + [target]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep only required fields
data = df[required_columns].copy()

# Clean categorical fields
for col in categorical_features + [target]:
    data[col] = (
        data[col]
        .astype("string")
        .str.strip()
    )

    data[col] = data[col].replace("", pd.NA)

# Convert numerical fields
for col in numerical_features:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# Remove rows without a target
data = data.dropna(subset=[target])

# Remove missing or invalid numeric targets
data = data.dropna(subset=numerical_features)

# Remove classes represented by fewer than 2 records.
# This is necessary for a stratified train/test split.
class_counts = data[target].value_counts()

rare_classes = class_counts[
    class_counts < 2
].index

if len(rare_classes) > 0:
    print("\nRare crime types excluded from training:")
    print(list(rare_classes))

    data = data[
        ~data[target].isin(rare_classes)
    ].copy()

print("\nRecords available for training:", len(data))

print("\nCrime type distribution:")
print(data[target].value_counts())


X = data[
    categorical_features + numerical_features
]

y = data[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))



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
    max_depth=12,
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


print("\nTraining crime type prediction model...")

model.fit(X_train, y_train)

print("Training completed successfully.")


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

print("\n======================================")
print(" MODEL PERFORMANCE")
print("======================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CPAPS CRIME TYPE CLASSIFICATION REPORT\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)


labels = sorted(y.unique())

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(12, 9))

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=True
)

plt.title("CPAPS - Crime Type Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MATRIX_FILE,
    dpi=300
)

plt.close()


model_bundle = {
    "model": model,
    "target": target,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "classes": list(model.named_steps["classifier"].classes_),
    "accuracy": accuracy
}

joblib.dump(
    model_bundle,
    MODEL_FILE
)


sample = X_test.iloc[[0]]

sample_prediction = model.predict(sample)[0]

print("\n======================================")
print(" SAMPLE PREDICTION")
print("======================================")

print("\nSample input:")
print(sample.to_string(index=False))

print("\nPredicted crime type:", sample_prediction)
print("Actual crime type:", y_test.iloc[0])


print("\n======================================")
print(" CRIME TYPE MODEL COMPLETED")
print("======================================")

print("\nFiles generated:")

print("1.", MODEL_FILE)
print("2.", REPORT_FILE)
print("3.", MATRIX_FILE)

print("\nCPAPS crime type prediction model saved successfully.")
