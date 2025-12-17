"""
Feature schema and data contracts for the Churn prediction project.

This module centralizes the expected structure of data at different
stages of the pipeline. At this stage, these definitions are used
primarily for documentation and consistency — not strict enforcement.

Pipeline stages:
- Raw data (as loaded from source CSV)
- Feature data (after preprocessing + feature engineering)
"""

# =============================================================================
# Target Variable
# =============================================================================

TARGET_COL = "Churn"

# =============================================================================
# Raw Data Schema (Telco Customer Churn dataset)
# =============================================================================

# Columns expected to exist in the raw dataset
RAW_REQUIRED_COLUMNS = {
    "customerID",
    "Churn",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "SeniorCitizen",
}

# Allowed categorical values in the raw dataset
# (Used for documentation and optional validation)
RAW_CATEGORICAL_DOMAINS = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

# =============================================================================
# Feature Schema (after preprocessing + feature engineering)
# =============================================================================

# Categorical features (kept as strings and one-hot encoded in the model)
CATEGORICAL_FEATURES = [
    "InternetService",
    "Contract",
    "PaymentMethod",
    "tenure_bucket",
]

# Binary features AFTER preprocessing (encoded as 0 / 1)
BINARY_FEATURES = [
    "gender",              # Male=1, Female=0
    "Partner",             # Yes=1, No=0
    "Dependents",          # Yes=1, No=0
    "PhoneService",        # Yes=1, No=0
    "MultipleLines",       # Yes=1, No=0
    "OnlineSecurity",      # Yes=1, No=0
    "OnlineBackup",        # Yes=1, No=0
    "DeviceProtection",    # Yes=1, No=0
    "TechSupport",         # Yes=1, No=0
    "StreamingTV",         # Yes=1, No=0
    "StreamingMovies",     # Yes=1, No=0
    "PaperlessBilling",    # Yes=1, No=0
    "SeniorCitizen",       # Already 0/1 in raw data
    "is_new_customer", 
]

# Numeric features (continuous)
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month", 
]

# Logical grouping of all expected model input columns (excluding target).
# NOTE: This is NOT the final model matrix order after one-hot encoding.
ALL_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES +
    BINARY_FEATURES +
    NUMERIC_FEATURES
)

# =============================================================================
#  Validation Thresholds
# =============================================================================

MIN_DATASET_SIZE = 1000

MIN_TENURE = 0
MAX_TENURE = 100  # months

MIN_MONTHLY_CHARGES = 0
MAX_MONTHLY_CHARGES = 200

MIN_TOTAL_CHARGES = 0
