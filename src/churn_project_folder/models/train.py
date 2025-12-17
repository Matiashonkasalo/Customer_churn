import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from churn_project_folder.models.model_registry import get_model_builder
from churn_project_folder.features.schema import (
    TARGET_COL,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES
)



def train_model(
    df: pd.DataFrame,
    model_name: str = "logistic",
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    log_model = False,
    **model_params,
):
    """
    Train a model specified by `model_name`.

    Assumes an active MLflow run exists.
    """

    # ---------------------------
    # 1. Split X / y
    # ---------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # ---------------------------
    # 2. Feature types
    # ---------------------------
    categorical_features = CATEGORICAL_FEATURES

    binary_features = BINARY_FEATURES

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
        and col not in binary_features
    ]



    # ---------------------------
    # 3. Preprocessor
    # ---------------------------
    preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
            categorical_features,
        ),
        (
            "bin",
            SimpleImputer(strategy="most_frequent"),
            binary_features,
        ),
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
    ]
    )


    # ---------------------------
    # 4. Build model via registry
    # ---------------------------
    model_builder = get_model_builder(model_name)
    model = model_builder(preprocessor, **model_params)

    # ---------------------------
    # 5. Log parameters
    # ---------------------------
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("num_numeric_features", len(numeric_features))
    mlflow.log_param("num_categorical_features", len(categorical_features))

    # ---------------------------
    # 6. Train
    # ---------------------------
    model.fit(X_train, y_train)

    # ---------------------------
    # 7. Log model artifact
    # ---------------------------
    if log_model: 

        mlflow.sklearn.log_model(
            sk_model=model,
            name = "model"
        )
        # ---------------------------
        # 8. Log training dataset
        # ---------------------------
        train_ds = from_pandas(
            X_train,
            source="local_pandas_dataframe",
        )
        mlflow.log_input(train_ds, context="training")

    return model, X_train, X_test, y_train, y_test
