import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model and log metrics to MLflow.
    """

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    return metrics
