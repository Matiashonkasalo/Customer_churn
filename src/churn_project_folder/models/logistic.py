from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def build_logistic_model(preprocessor, **model_params):
    defaults = dict(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )

    defaults.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(**defaults)),
        ]
    )

