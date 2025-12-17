from churn_project_folder.models.logistic import build_logistic_model
from churn_project_folder.models.random_forest import build_random_forest_model
from churn_project_folder.models.xgboost import build_xgboost_model

MODEL_BUILDERS = {
    "logistic": build_logistic_model,
    "random_forest": build_random_forest_model,
    "xgboost": build_xgboost_model,
}


def get_model_builder(model_name: str):
    """
    Return the model builder function for the given model name.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_BUILDERS.keys())}"
        )
    return MODEL_BUILDERS[model_name]
