from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def build_xgboost_model(preprocessor, **model_params):
    """
    Build an XGBoost classification pipeline.
    """

    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )

    # Override defaults ONLY if tuning passes params
    defaults.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**defaults)),
        ]
    )
