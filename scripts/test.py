import mlflow.sklearn
from pathlib import Path

EXPORT_PATH = Path("src/churn_project_folder/serving/models/churn_model")

model = mlflow.sklearn.load_model(
    "models:/churn_model@production"
)

mlflow.sklearn.save_model(model, EXPORT_PATH)

print(f"Model exported to {EXPORT_PATH}")
