import optuna
import mlflow
from churn_project_folder.models.train import train_model
from churn_project_folder.models.evaluate import evaluate_model


def tune_logistic(
    df,
    n_trials: int = 20,
    metric: str = "roc_auc",
):
    """
    Hyperparameter tuning for Logistic Regression using Optuna.

    Assumes an active MLflow run (parent).
    """

    def objective(trial):
        # --------------------------------------------------
        # 1. Sample hyperparameters
        # --------------------------------------------------
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": "lbfgs",
        }

        # --------------------------------------------------
        # 2. Nested MLflow run (one per trial)
        # --------------------------------------------------
        with mlflow.start_run(
            nested=True,
            run_name=f"trial_{trial.number}"
        ):

            for k, v in params.items():
                mlflow.log_param(k, v)

            # --------------------------------------------------
            # 3. Train
            # --------------------------------------------------
            model, X_train, X_test, y_train, y_test = train_model(
                df,
                model_name="logistic",
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
    study = optuna.create_study(direction="maximize", study_name="logistic_tuning")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
