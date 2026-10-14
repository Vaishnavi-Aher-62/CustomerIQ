
from pathlib import Path
import joblib
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "training_snapshots.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.joblib"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load saved model and feature definitions
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

data = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

train = data[data["Split"] == "train"].copy()
test = data[data["Split"] == "test"].copy()

X_background = train[feature_columns]
X_test = test[feature_columns]

# Apply the SAME preprocessing fitted inside the saved pipeline.
preprocessor = model[:-1]
classifier = model.named_steps["logisticregression"]

background_transformed = preprocessor.transform(X_background)
test_transformed = preprocessor.transform(X_test)

# Explain Logistic Regression using SHAP's linear explainer.
explainer = shap.LinearExplainer(
    classifier,
    background_transformed
)

shap_values = explainer(test_transformed)

# Binary Logistic Regression explanations are in log-odds space.
values = shap_values.values

if values.ndim == 3:
    values = values[:, :, 1]

# Global explanation: average absolute contribution per feature.
global_importance = (
    pd.Series(
        abs(values).mean(axis=0),
        index=feature_columns,
        name="MeanAbsoluteSHAP"
    )
    .sort_values(ascending=False)
    .reset_index()
)

global_importance.columns = ["Feature", "MeanAbsoluteSHAP"]

global_importance.to_csv(
    REPORT_DIR / "shap_global_importance.csv",
    index=False
)

# Individual explanations for the highest-risk test customers.
predictions["CustomerID"] = predictions["CustomerID"].astype(str)
test["CustomerID"] = test["CustomerID"].astype(str)

explanation_data = predictions.merge(
    test[["CustomerID", "SnapshotDate"]],
    on=["CustomerID", "SnapshotDate"],
    how="left",
    validate="one_to_one"
)

explanation_data = explanation_data.sort_values(
    "ChurnRisk", ascending=False
).reset_index(drop=True)

# Map saved predictions to their corresponding test rows.
test_keys = test[["CustomerID", "SnapshotDate"]].reset_index(drop=True)
prediction_keys = predictions[["CustomerID", "SnapshotDate"]].reset_index(drop=True)

key_to_row = {
    (row.CustomerID, row.SnapshotDate): i
    for i, row in test_keys.iterrows()
}

top_keys = list(zip(
    explanation_data["CustomerID"],
    explanation_data["SnapshotDate"]
))[:100]

selected_rows = [
    key_to_row[key] for key in top_keys if key in key_to_row
]

contributions = pd.DataFrame(
    values[selected_rows],
    columns=feature_columns
)

contributions.insert(
    0,
    "CustomerID",
    test.iloc[selected_rows]["CustomerID"].to_numpy()
)

contributions.insert(
    1,
    "SnapshotDate",
    test.iloc[selected_rows]["SnapshotDate"].to_numpy()
)

contributions.to_csv(
    REPORT_DIR / "shap_customer_explanations.csv",
    index=False
)

print("\n--- GLOBAL SHAP IMPORTANCE ---")
print(global_importance.round(4).to_string(index=False))

print("\nSaved:")
print("reports/shap_global_importance.csv")
print("reports/shap_customer_explanations.csv")

print(
    "\nInterpretation: positive SHAP values increase the model's "
    "log-odds of churn; negative values decrease them. "
    "These are model explanations, not causal effects."
)
