"""Core prediction logic shared by the FastAPI API and the Streamlit app.

It repeats exactly the preprocessing steps done in the notebook:
1. One-hot encoding (Month, VisitorType)
2. Feature engineering (TotalPages, TotalDuration, AvgTimePerPage)
3. Capping at the 99th percentile (limits saved from the training set)
4. StandardScaler
5. Model prediction
"""
import os

import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.joblib"))

_cap_path = os.path.join(MODEL_DIR, "cap_limits.joblib")
cap_limits = joblib.load(_cap_path) if os.path.exists(_cap_path) else None


def build_row(data: dict) -> pd.DataFrame:
    """Turn the visitor's raw inputs into one row with the same columns as the training data."""
    row = pd.DataFrame(0.0, index=[0], columns=columns)

    # 1) numeric inputs
    for key, value in data.items():
        if key in ("Month", "VisitorType"):
            continue
        if key in row.columns:
            row.loc[0, key] = float(value)

    # 2) one-hot columns (the first category was dropped in training, so it stays all zeros)
    month_col = "Month_" + str(data["Month"])
    if month_col in row.columns:
        row.loc[0, month_col] = 1.0

    visitor_col = "VisitorType_" + str(data["VisitorType"])
    if visitor_col in row.columns:
        row.loc[0, visitor_col] = 1.0

    # 3) engineered features (same formulas as in the notebook)
    total_pages = data["Administrative"] + data["Informational"] + data["ProductRelated"]
    total_duration = (data["Administrative_Duration"] + data["Informational_Duration"]
                      + data["ProductRelated_Duration"])
    row.loc[0, "TotalPages"] = float(total_pages)
    row.loc[0, "TotalDuration"] = float(total_duration)
    row.loc[0, "AvgTimePerPage"] = float(total_duration / total_pages) if total_pages > 0 else 0.0

    # 4) capping with the limits learned on the training set
    if cap_limits is not None:
        for col in cap_limits.index:
            if col in row.columns:
                row.loc[0, col] = min(row.loc[0, col], float(cap_limits[col]))

    return row


def predict(data: dict) -> dict:
    row = build_row(data)
    X = scaler.transform(row)
    probability = float(model.predict_proba(X)[0, 1])
    will_buy = bool(model.predict(X)[0] == 1)
    return {
        "will_buy": will_buy,
        "probability": round(probability, 4),
        "label": "Likely to buy" if will_buy else "Unlikely to buy",
    }
