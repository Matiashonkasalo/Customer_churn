import optuna
import mlflow

from churn_project_folder.models.train import train_model
from churn_project_folder.models.evaluate import evaluate_model


def tune_xgboost(
    df,
    n_trials: int = 20,
    metric: str = "roc_auc",
):
    """
    Hyperparameter tuning for XGBoost using Optuna.

    Assumes an active MLflow run (parent).
    """

    def objective(trial):
        # --------------------------------------------------
        # 1. Sample hyperparameters
        # --------------------------------------------------
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }

        # --------------------------------------------------
        # 2. Nested MLflow run (one trial = one run)
        # --------------------------------------------------
        with mlflow.start_run(nested=True):
            for k, v in params.items():
                mlflow.log_param(k, v)

            # --------------------------------------------------
            # 3. Train model (reuse existing pipeline)
            # --------------------------------------------------
            model, X_train, X_test, y_train, y_test = train_model(
                df,
                model_name="xgboost",
                **params,
            )

            # --------------------------------------------------
            # 4. Evaluate model
            # --------------------------------------------------
            metrics = evaluate_model(model, X_test, y_test)

            score = metrics.get(metric)
            if score is None:
                raise ValueError(
                    f"Metric '{metric}' not found in evaluation results"
                )

            return score

    # ------------------------------------------------------
    # 5. Run Optuna study
    # ------------------------------------------------------
    study = optuna.create_study(direction="maximize",study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
