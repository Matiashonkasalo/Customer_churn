import optuna
import mlflow

from churn_project_folder.models.train import train_model
from churn_project_folder.models.evaluate import evaluate_model


def tune_random_forest(
    df,
    n_trials: int = 20,
    metric: str = "roc_auc",
):
    """
    Hyperparameter tuning for Random Forest using Optuna.
    Assumes an active MLflow run (parent).
    """

    def objective(trial):
        # --------------------------------------------------
        # 1. Sample hyperparameters
        # --------------------------------------------------
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

        # --------------------------------------------------
        # 2. Nested MLflow run (one per trial)
        # --------------------------------------------------
        with mlflow.start_run(nested=True):
            for k, v in params.items():
                mlflow.log_param(k, v)

            # --------------------------------------------------
            # 3. Train
            # --------------------------------------------------
            model, X_train, X_test, y_train, y_test = train_model(
                df,
                model_name="random_forest",
                **params,
            )

            # --------------------------------------------------
            # 4. Evaluate
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
    study = optuna.create_study(direction="maximize", study_name="random_forest_tuning")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
