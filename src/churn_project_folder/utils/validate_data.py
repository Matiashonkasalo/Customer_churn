import pandas as pd

from churn_project_folder.features.schema import (
    RAW_CATEGORICAL_DOMAINS,
    MIN_DATASET_SIZE,
)


def validate_raw_telco_data(df: pd.DataFrame) -> None:
    """
    Validate raw Telco churn data before preprocessing.
    Checks schema, keys, and categorical semantics.
    """

    # ------------------------------------------------------------------
    # Required columns (minimal gatekeeper)
    # ------------------------------------------------------------------
    required_columns = {
        "customerID",
        "Churn",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "gender",
        "Partner",
    }

    # NOTE: We intentionally keep this minimal set to avoid breaking
    # existing pipeline behavior.
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Business key integrity
    # ------------------------------------------------------------------
    if df["customerID"].isnull().any():
        raise ValueError("Null values found in customerID")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found")

    # ------------------------------------------------------------------
    # Target validity
    # ------------------------------------------------------------------
    allowed_churn_values = RAW_CATEGORICAL_DOMAINS["Churn"]
    if not set(df["Churn"].dropna().unique()).issubset(allowed_churn_values):
        raise ValueError("Invalid values found in Churn")

    # ------------------------------------------------------------------
    # Categorical domains used in preprocessing
    # ------------------------------------------------------------------
    allowed_gender_values = RAW_CATEGORICAL_DOMAINS["gender"]
    if not set(df["gender"].dropna().unique()).issubset(allowed_gender_values):
        raise ValueError("Invalid values found in gender")

    allowed_partner_values = RAW_CATEGORICAL_DOMAINS["Partner"]
    if not set(df["Partner"].dropna().unique()).issubset(allowed_partner_values):
        raise ValueError("Invalid values found in Partner")

    # ------------------------------------------------------------------
    # Volume sanity
    # ------------------------------------------------------------------
    if len(df) < MIN_DATASET_SIZE:
        raise ValueError("Dataset too small for modeling")

