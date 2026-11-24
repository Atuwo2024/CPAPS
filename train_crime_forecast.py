
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_outputs")

DATA_FILE = os.path.join(OUTPUT_DIR, "cleaned_crime_data.csv")

MODEL_FILE = os.path.join(OUTPUT_DIR, "crime_forecast_model.pkl")
FORECAST_FILE = os.path.join(OUTPUT_DIR, "crime_forecast_next_12_months.csv")
EVALUATION_FILE = os.path.join(OUTPUT_DIR, "forecast_model_evaluation.csv")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "monthly_crime_history.csv")
CHART_FILE = os.path.join(OUTPUT_DIR, "crime_forecast_chart.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n======================================")
print(" CPAPS - CRIME FORECASTING")
print("======================================")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check that cleaned_crime_data.csv is inside ml_outputs."
    )

df = pd.read_csv(DATA_FILE)

# Standardise column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCrime dataset loaded successfully.")
print("Total records:", len(df))
print("Available columns:", list(df.columns))

# Identify the incident date column
date_candidates = ["incident_date", "date", "crime_date"]

date_column = next(
    (col for col in date_candidates if col in df.columns),
    None
)

if date_column is None:
    raise ValueError(
        "Could not find the incident date column. "
        "Expected incident_date, date, or crime_date."
    )

# Convert dates
df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df = df.dropna(subset=[date_column]).copy()

if df.empty:
    raise ValueError("No valid incident dates were found.")


df["month_start"] = (
    df[date_column]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly = (
    df.groupby("month_start")
    .size()
    .rename("crime_count")
    .sort_index()
)

# Fill missing months between the first and last
# recorded months with zero recorded incidents.
all_months = pd.date_range(
    start=monthly.index.min(),
    end=monthly.index.max(),
    freq="MS"
)

monthly = monthly.reindex(all_months, fill_value=0)
monthly.index.name = "month"

history = monthly.reset_index()

history.to_csv(HISTORY_FILE, index=False)

print("\nMonthly crime history created.")
print("First month:", monthly.index.min().strftime("%B %Y"))
print("Last month:", monthly.index.max().strftime("%B %Y"))
print("Number of months:", len(monthly))

if len(monthly) < 30:
    raise ValueError(
        "There are too few monthly observations for this "
        "forecasting experiment. At least 30 are required."
    )


def create_features(series):
    """
    Create features using only previous months.
    """

    data = pd.DataFrame(index=series.index)

    data["lag_1"] = series.shift(1)
    data["lag_2"] = series.shift(2)
    data["lag_3"] = series.shift(3)
    data["lag_12"] = series.shift(12)

    data["rolling_mean_3"] = (
        series.shift(1)
        .rolling(window=3)
        .mean()
    )

    data["month_number"] = series.index.month

    data["time_index"] = np.arange(len(series))

    return data


features = create_features(monthly)

model_data = features.copy()
model_data["crime_count"] = monthly

model_data = model_data.dropna()

X = model_data.drop(columns=["crime_count"])
y = model_data["crime_count"]


# Keep the latest 12 available months for testing.
test_size = 12

if len(model_data) <= test_size:
    raise ValueError(
        "Not enough observations remain after creating "
        "lag features for a 12-month test."
    )

X_train = X.iloc[:-test_size]
X_test = X.iloc[-test_size:]

y_train = y.iloc[:-test_size]
y_test = y.iloc[-test_size:]

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

print(
    "Testing period:",
    X_test.index.min().strftime("%B %Y"),
    "to",
    X_test.index.max().strftime("%B %Y")
)


rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = np.maximum(
    rf_model.predict(X_test),
    0
)

# Compare each test month with the same month
# in the previous year.

baseline_predictions = monthly.reindex(
    X_test.index - pd.DateOffset(years=1)
)

if baseline_predictions.isna().any():
    raise ValueError(
        "Could not calculate the seasonal baseline "
        "for every test month."
    )

baseline_predictions = baseline_predictions.to_numpy()



def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    return mae, rmse


rf_mae, rf_rmse = calculate_metrics(
    y_test,
    rf_predictions
)

baseline_mae, baseline_rmse = calculate_metrics(
    y_test,
    baseline_predictions
)

evaluation = pd.DataFrame([
    {
        "Model": "Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse
    },
    {
        "Model": "Seasonal Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse
    }
])

evaluation.to_csv(EVALUATION_FILE, index=False)

print("\n======================================")
print(" MODEL EVALUATION")
print("======================================")

print(evaluation.to_string(index=False))

# Choose the approach with the lower MAE.
if rf_mae <= baseline_mae:
    selected_method = "Random Forest"
    selected_model = rf_model

    print("\nSelected model: Random Forest")

else:
    selected_method = "Seasonal Baseline"
    selected_model = None

    print("\nSelected approach: Seasonal Baseline")

print(
    "\nMAE measures the average absolute prediction error "
    "in number of incidents."
)


future_months = pd.date_range(
    start=monthly.index.max() + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)

extended_series = monthly.copy()
forecast_values = []

for future_date in future_months:

    if selected_method == "Random Forest":

        next_features = pd.DataFrame(
            [{
                "lag_1": extended_series.iloc[-1],
                "lag_2": extended_series.iloc[-2],
                "lag_3": extended_series.iloc[-3],
                "lag_12": extended_series.iloc[-12],
                "rolling_mean_3": extended_series.iloc[-3:].mean(),
                "month_number": future_date.month,
                "time_index": len(extended_series)
            }]
        )

        prediction = selected_model.predict(
            next_features
        )[0]

    else:
        # Use the same month from the previous year.
        previous_year_date = (
            future_date - pd.DateOffset(years=1)
        )

        prediction = extended_series.loc[
            previous_year_date
        ]

    prediction = max(0, float(prediction))

    forecast_values.append(prediction)

    # Add prediction so the next month can use it.
    extended_series.loc[future_date] = prediction


forecast = pd.DataFrame({
    "month": future_months,
    "predicted_crime_count": [
        round(value) for value in forecast_values
    ],
    "forecast_method": selected_method
})

forecast.to_csv(FORECAST_FILE, index=False)

# Save model and information for later integration.
model_bundle = {
    "selected_method": selected_method,
    "model": selected_model,
    "feature_columns": list(X.columns),
    "historical_start": str(monthly.index.min().date()),
    "historical_end": str(monthly.index.max().date()),
    "forecast_months": 12,
    "evaluation": evaluation.to_dict(orient="records")
}

joblib.dump(model_bundle, MODEL_FILE)


plt.figure(figsize=(13, 6))

plt.plot(
    monthly.index,
    monthly.values,
    label="Historical Crime Records",
    marker="o"
)

plt.plot(
    future_months,
    forecast_values,
    label="Forecasted Crime Counts",
    marker="o",
    linestyle="--"
)

plt.axvline(
    monthly.index.max(),
    linestyle=":",
    label="Forecast Begins"
)

plt.title("CPAPS - Monthly Crime Forecast")
plt.xlabel("Month")
plt.ylabel("Number of Recorded Crimes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(CHART_FILE, dpi=300)
plt.close()


print("\n======================================")
print(" FORECASTING COMPLETED")
print("======================================")

print("\nNext 12 months forecast:")

print(forecast.to_string(index=False))

print("\nFiles saved successfully:")

print("1.", MODEL_FILE)
print("2.", FORECAST_FILE)
print("3.", EVALUATION_FILE)
print("4.", HISTORY_FILE)
print("5.", CHART_FILE)

print("\nCPAPS forecasting preparation completed.")
