from churn_project_folder.data.load_data import load_raw_data
from churn_project_folder.utils.validate_data import validate_raw_telco_data
from churn_project_folder.data.preprocess import preprocess_data
from churn_project_folder.features.build_features import build_features
import mlflow
from churn_project_folder.models.train import train_model
from churn_project_folder.models.evaluate import evaluate_model
from churn_project_folder.models.tune_logistic import tune_logistic
from churn_project_folder.models.tune_xgboost import tune_xgboost
from churn_project_folder.models.tune_random_forest import tune_random_forest
from churn_project_folder.models.model_registry import MODEL_BUILDERS
from churn_project_folder.features.schema import (
    TARGET_COL,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)

TUNERS = {
    "logistic": tune_logistic,
    "random_forest": tune_random_forest,
    "xgboost": tune_xgboost,
}


ENABLE_TUNING = True



def _check_feature_contract(df):
    # ------------------------------------------------------------------
    # 1. Target must exist
    # ------------------------------------------------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' missing after build_features"
        )

    # ------------------------------------------------------------------
    # 2. No unexpected object columns
    # ------------------------------------------------------------------
    object_cols = (
        df.drop(columns=[TARGET_COL])
        .select_dtypes(include="object")
        .columns
    )
    unexpected = set(object_cols) - set(CATEGORICAL_FEATURES)
    if unexpected:
        raise ValueError(
            f"Unexpected object columns after build_features: {unexpected}"
        )

    # ------------------------------------------------------------------
    # 3. Binary columns sanity (0 / 1 only)
    # ------------------------------------------------------------------
    for col in BINARY_FEATURES:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    f"Column '{col}' is not binary after build_features: "
                    f"{unique_vals}"
                )

    # ------------------------------------------------------------------
    # 4. Missing-value diagnostics (THIS IS THE KEY)
    # ------------------------------------------------------------------
    binary_nan_counts = {}
    for col in BINARY_FEATURES:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                binary_nan_counts[col] = nan_count

    if binary_nan_counts:
        msg = (
            "Unexpected NaNs found in binary feature columns after "
            "build_features:\n"
        )
        for col, count in binary_nan_counts.items():
            msg += f"  - {col}: {count} NaNs\n"

        raise ValueError(msg)
    
def test_data():
    df_raw = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    validate_raw_telco_data(df_raw)
    df_clean = preprocess_data(df_raw)
    df_features = build_features(df_clean)
    _check_feature_contract(df_features)
    print(df_features['tenure'])
    print(df_features['tenure_bucket'])
    print(df_features['is_new_customer'])
    print(df_features['charges_per_month'])
    print(df_features.isna().sum().sort_values(ascending=False).head())

    

def run_pipeline(data_path: str, model_name: str = "logistic", run_all_models: bool = False):  
    if run_all_models:
        models_to_run = list(MODEL_BUILDERS.keys())
    else:
        if model_name not in MODEL_BUILDERS:
            raise ValueError(
                f"Invalid model_name '{model_name}'. "
                f"Available models: {list(MODEL_BUILDERS.keys())}"
            )
        models_to_run = [model_name]
    
    # --------------------------------------------------
    # 1. Load & prepare data 
    # --------------------------------------------------
    df_raw = load_raw_data(data_path)
    validate_raw_telco_data(df_raw)
    df_clean = preprocess_data(df_raw)
    df_features = build_features(df_clean)
    _check_feature_contract(df_features)
    
    # --------------------------------------------------
    # 2. Loop over models
    # --------------------------------------------------

    for model in models_to_run:

        # ---------- Baseline ----------
        with mlflow.start_run(run_name=f"{model}_baseline"):
            trained_model, X_train, X_test, y_train, y_test = train_model(
                df_features,
                model_name=model,
                log_model = ENABLE_TUNING
            )

            metrics = evaluate_model(trained_model, X_test, y_test)

            print(f"\n{model.upper()} baseline metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")


        # ---------- Tuning ----------
        if ENABLE_TUNING and model in TUNERS:
            with mlflow.start_run(run_name=f"{model}_tuning"):

                best_params, best_score = TUNERS[model](
                    df_features,
                    n_trials=20,
                    metric="roc_auc",
                )

                # log best score as METRIC (not param)
                mlflow.log_metric("best_score", best_score)

                # log best hyperparameters
                for k, v in best_params.items():
                    mlflow.log_param(f"best_{k}", v)

                # train & log the BEST model
                best_model, *_ = train_model(
                    df_features,
                    model_name=model,
                    **best_params,
                )

                mlflow.sklearn.log_model(
                    best_model,
                    name="best_model"
                )


            

if __name__ == "__main__":
    #tune parameter (logistic, random_forest, xgboost)/ set run_all_models = True to run every model
    run_pipeline("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", run_all_models=True)
    #test_data()