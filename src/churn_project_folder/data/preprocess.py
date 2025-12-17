import pandas as pd


def check_rows(df):
    #function that deletes a row if there are 90 or more % of data missing
    before_rows = len(df)
    ROW_NULL_THRESHOLD = 0.9  # drop rows with ≥90% missing
    row_null_frac = df.isnull().mean(axis=1)
    df = df.loc[row_null_frac < ROW_NULL_THRESHOLD]
    dropped = before_rows - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with ≥{ROW_NULL_THRESHOLD:.0%} missing values")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for Telco churn data.
    """
    df = df.copy()

    # Strip column names
    df.columns = df.columns.str.strip()

    df = check_rows(df)

    # Drop customer ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Replace 'No internet service' with 'No'
    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

    # Replace 'No phone service' with 'No'
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Missing TotalCharges typically occur for tenure = 0
    # Median imputation is stable and avoids leakage

    # Encode target
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        if df["Churn"].isna().any():
            raise ValueError("Unexpected values found in Churn column")

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)


    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")


    return df
