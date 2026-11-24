
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_outputs")

files = [
    "crime_forecast_model.pkl",
    "crime_type_model.pkl",
    "crime_risk_model.pkl"
]

print("\n========== CPAPS MODEL INSPECTION ==========\n")

for filename in files:

    path = os.path.join(MODEL_DIR, filename)

    print(f"\nMODEL: {filename}")
    print("-" * 45)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    try:
        bundle = joblib.load(path)

        if isinstance(bundle, dict):

            print("Bundle contents:")

            for key, value in bundle.items():

                if hasattr(value, "shape"):
                    print(f"{key}: shape = {value.shape}")

                elif isinstance(value, (list, tuple)):
                    print(f"{key}: {value}")

                elif isinstance(value, dict):
                    print(f"{key}: {value}")

                elif key == "model":
                    print(f"{key}: {type(value).__name__}")

                else:
                    print(f"{key}: {value}")

        else:
            print("Object type:", type(bundle).__name__)

    except Exception as e:
        print("Inspection error:", e)

print("\n========== INSPECTION COMPLETE ==========\n")
