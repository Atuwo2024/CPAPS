import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

sns.set_theme(style="darkgrid")
OUTPUT_DIR = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  DB CONNECTION 
def get_connection():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = "12345",   
        database = "cpaps_db"
    )

#  LOAD & PREPARE DATA 
def load_and_prepare():
    print("📦 Loading data from database...")
    conn  = get_connection()
    df    = pd.read_sql("SELECT * FROM v_crime_report", conn)
    conn.close()
    print(f"✅ Loaded {len(df)} records\n")

    # Feature Engineering
    df['hour_of_day']    = df['hour_of_day'].fillna(0).astype(int)
    df['month_number']   = df['month_number'].astype(int)
    df['year_number']    = df['year_number'].astype(int)
    df['fatalities']     = df['fatalities'].astype(int)
    df['injuries_reported'] = df['injuries_reported'].astype(int)
    df['property_loss_ksh'] = df['property_loss_ksh'].fillna(0).astype(float)

    # Encode categorical columns
    le = LabelEncoder()
    df['county_enc']       = le.fit_transform(df['county'])
    df['area_type_enc']    = le.fit_transform(df['area_type'])
    df['day_enc']          = le.fit_transform(df['day_of_week'])
    df['severity_enc']     = le.fit_transform(df['severity_level'])
    df['weather_enc']      = le.fit_transform(df['weather_condition']) if 'weather_condition' in df.columns else 0
    df['category_enc']     = le.fit_transform(df['crime_category'])
    df['crime_type_enc']   = le.fit_transform(df['crime_type'])

    # Save label mappings for display
    crime_type_classes = le.classes_

    print("Features ready:")
    print(f"   Columns available: {list(df.columns)}\n")
    return df, crime_type_classes

#  FEATURES & TARGETS 
def get_features(df):
    features = [
        'hour_of_day', 'month_number', 'year_number',
        'county_enc', 'area_type_enc', 'day_enc',
        'severity_enc', 'property_loss_ksh',
        'injuries_reported', 'fatalities'
    ]
    return df[features]

# MODEL 1: RANDOM FOREST — Predict Crime Type

def random_forest_model(df, crime_type_classes):
    print("=" * 55)
    print("  MODEL 1: RANDOM FOREST — Crime Type Prediction")
    print("=" * 55)

    X = get_features(df)
    y = df['crime_type_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")

    # Cross validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Feature Importance Chart
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='steelblue')
    plt.title('Random Forest — Feature Importance', fontsize=15, fontweight='bold')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_feature_importance.png")
    plt.show()
    print("✅ Feature importance chart saved")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=crime_type_classes,
                yticklabels=crime_type_classes)
    plt.title('Random Forest — Confusion Matrix', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix saved")

    # Save model
    joblib.dump(model, f"{OUTPUT_DIR}/random_forest_model.pkl")
    print("✅ Model saved: random_forest_model.pkl\n")

    return model, acc

# MODEL 2: LOGISTIC REGRESSION — Predict High Severity Crime
# 
def logistic_regression_model(df):
    print("=" * 55)
    print("  MODEL 2: LOGISTIC REGRESSION — High Severity Prediction")
    print("=" * 55)

    X = get_features(df)

    # Binary target: 1 = Critical/High severity, 0 = Medium/Low
    y = df['severity_level'].apply(
        lambda x: 1 if x in ['Critical', 'High'] else 0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")

    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low/Medium', 'High/Critical']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low/Medium', 'High/Critical'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Oranges', colorbar=False)
    plt.title('Logistic Regression — Confusion Matrix', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lr_confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix saved")

    # Probability Distribution Chart
    plt.figure(figsize=(10, 5))
    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='green',  label='Low/Medium Severity')
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='red',    label='High/Critical Severity')
    plt.title('Logistic Regression — Predicted Probability Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability of High/Critical Severity')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lr_probability_distribution.png")
    plt.show()
    print("✅ Probability distribution chart saved")

    joblib.dump(model,  f"{OUTPUT_DIR}/logistic_regression_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/lr_scaler.pkl")
    print("✅ Model saved: logistic_regression_model.pkl\n")

    return model, acc

# 
# MODEL 3: DECISION TREE — Visual & Interpretable
# 
def decision_tree_model(df, crime_type_classes):
    print("=" * 55)
    print("  MODEL 3: DECISION TREE — Crime Category Prediction")
    print("=" * 55)

    X = get_features(df)
    y = df['category_enc']

    le = LabelEncoder()
    category_classes = le.fit_transform(df['crime_category'].unique())
    category_names   = df['crime_category'].unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Decision Tree Visualization
    plt.figure(figsize=(24, 10))
    plot_tree(
        model,
        feature_names=X.columns.tolist(),
        class_names=[str(c) for c in category_names],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.title('Decision Tree — Crime Category Prediction (Max Depth 3)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/decision_tree_visual.png", dpi=150)
    plt.show()
    print("✅ Decision tree visual saved")

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    importances.plot(kind='bar', color='teal')
    plt.title('Decision Tree — Feature Importance', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dt_feature_importance.png")
    plt.show()
    print("✅ Decision tree feature importance saved")

    joblib.dump(model, f"{OUTPUT_DIR}/decision_tree_model.pkl")
    print("✅ Model saved: decision_tree_model.pkl\n")

    return model, acc

# 
# MODELS COMPARISON SUMMARY
# 
def plot_model_comparison(rf_acc, lr_acc, dt_acc):
    models      = ['Random Forest', 'Logistic Regression', 'Decision Tree']
    accuracies  = [rf_acc * 100, lr_acc * 100, dt_acc * 100]
    colors      = ['steelblue', 'darkorange', 'teal']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors, width=0.5)
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc:.2f}%',
            ha='center', fontweight='bold', fontsize=12
        )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png")
    plt.show()
    print("✅ Model comparison chart saved")

    print("\n" + "=" * 55)
    print("         CPAPS — ML MODELS SUMMARY")
    print("=" * 55)
    print(f"  Random Forest Accuracy      : {rf_acc * 100:.2f}%")
    print(f"  Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    print(f"  Decision Tree Accuracy      : {dt_acc * 100:.2f}%")
    best = models[accuracies.index(max(accuracies))]
    print(f"\n  🏆 Best Model: {best} ({max(accuracies):.2f}%)")
    print("=" * 55)
    print(f"\n🎉 All models saved in '{OUTPUT_DIR}/' folder!")

# ── MAIN 
if __name__ == "__main__":
    print("\n🤖 Starting Machine Learning Model Training...\n")
    df, crime_type_classes = load_and_prepare()
    rf_model,  rf_acc  = random_forest_model(df, crime_type_classes)
    lr_model,  lr_acc  = logistic_regression_model(df)
    dt_model,  dt_acc  = decision_tree_model(df, crime_type_classes)
    plot_model_comparison(rf_acc, lr_acc, dt_acc)
    print("\n✅ Phase 4 Complete — ML Models built, evaluated and saved!")
