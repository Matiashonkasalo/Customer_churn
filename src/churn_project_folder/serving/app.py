from fastapi import FastAPI
from churn_project_folder.serving.inference import predict_from_raw
from churn_project_folder.serving.schemas import PredictRequest
from churn_project_folder.serving.gradio_app import create_gradio_app
import gradio as gr

app = FastAPI(title="Churn Prediction API")

#app.include_router(router)
#mount_ui(app)


@app.get("/")
def root_endpoint():
    return {"message": "API is running!"}

# run this uvicorn src.churn_project_folder.serving.app:app --reload


@app.post("/predict")
def predict(request: PredictRequest):
    return predict_from_raw(request.model_dump())


gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")