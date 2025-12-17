from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_random_forest_model(preprocessor, **model_params):
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    defaults.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**defaults)),
        ]
    )
