import pandas as pd


def build_features(df):
    # after feature engineering

     # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6", "6-12", "12-24", "24-48", "48+"],
        include_lowest=True,
    )

    # New customer flag
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

   
    # Charges per month
    df["charges_per_month"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    )
    return df
