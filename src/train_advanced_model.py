
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "training_snapshots.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["SnapshotDate"])

FEATURES = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue"
]

train = df[df["Split"] == "train"].copy()
valid = df[df["Split"] == "validation"].copy()
test = df[df["Split"] == "test"].copy()

if min(len(train), len(valid), len(test)) == 0:
    raise ValueError("Training, validation, or test split is empty.")

X_train, y_train = train[FEATURES], train["Churn"]
X_valid, y_valid = valid[FEATURES], valid["Churn"]
X_test, y_test = test[FEATURES], test["Churn"]

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}

validation_results = {}

# Select model using validation data only
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_valid)[:, 1]

    validation_results[name] = {
        "model": model,
        "roc_auc": roc_auc_score(y_valid, probs),
        "average_precision": average_precision_score(y_valid, probs)
    }

print("\n--- VALIDATION RESULTS ---")
for name, result in validation_results.items():
    print(
        f"{name}: ROC-AUC={result['roc_auc']:.4f}, "
        f"Average Precision={result['average_precision']:.4f}"
    )

best_name = max(
    validation_results,
    key=lambda name: validation_results[name]["roc_auc"]
)
best_model = validation_results[best_name]["model"]

# Final evaluation on the later test snapshot
test_probs = best_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_results = {
    "selected_model": best_name,
    "validation_roc_auc": validation_results[best_name]["roc_auc"],
    "test_roc_auc": roc_auc_score(y_test, test_probs),
    "test_average_precision": average_precision_score(y_test, test_probs),
    "test_accuracy": float((test_preds == y_test).mean()),
    "test_cutoff": str(test["SnapshotDate"].iloc[0].date()),
    "test_customers": int(len(test))
}

print("\n--- FINAL TEST RESULTS ---")
print(json.dumps(test_results, indent=2))
print("\nClassification report:")
print(classification_report(y_test, test_preds, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_preds))

# Save selected model and metadata
joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")
joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")

with open(REPORT_DIR / "model_metrics.json", "w") as f:
    json.dump(test_results, f, indent=2)

predictions = test[["CustomerID", "SnapshotDate", "Churn"]].copy()
predictions["ChurnRisk"] = test_probs
predictions["PredictedChurn"] = test_preds
predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

print("\nSaved model, metrics, and test predictions.")
