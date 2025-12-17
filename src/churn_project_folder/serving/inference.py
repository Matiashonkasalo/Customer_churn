import pandas as pd
from typing import Dict, Any
import mlflow.sklearn
from pathlib import Path

from churn_project_folder.data.preprocess import preprocess_data
from churn_project_folder.features.build_features import build_features
from churn_project_folder.features.schema import ALL_FEATURE_COLUMNS

# --------------------------------------------------
# Load model ONCE at startup
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "churn_model"
model = mlflow.sklearn.load_model(MODEL_PATH)


def predict_from_raw(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    # 1️ Raw input → DataFrame
    df_raw = pd.DataFrame([raw_input])

    # 2️ Preprocess + feature engineering
    df_processed = preprocess_data(df_raw)
    df_features = build_features(df_processed)

    # 3️ Enforce feature contract
    missing = set(ALL_FEATURE_COLUMNS) - set(df_features.columns)
    if missing:
        raise ValueError(f"Missing features at inference time: {missing}")

    # 4️ Align column order
    X = df_features[ALL_FEATURE_COLUMNS]

    # 5️ Predict
    churn_prob = model.predict_proba(X)[0, 1]
    prediction = int(churn_prob >= 0.5)

    return {
        "prediction": prediction,
        "churn_probability": float(churn_prob),
    }
