import gradio as gr
from churn_project_folder.serving.inference import predict_from_raw


def gradio_predict(
    tenure,
    MonthlyCharges,
    TotalCharges,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
):
    raw_input = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
    }

    result = predict_from_raw(raw_input)

    return result["prediction"], result["churn_probability"]


def create_gradio_app():
    return gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Number(label="Tenure (months)", value=12),
            gr.Number(label="Monthly Charges", value=70.5),
            gr.Number(label="Total Charges", value=845.0),

            gr.Radio(["Male", "Female"], label="Gender"),
            gr.Radio([0, 1], label="Senior Citizen"),

            gr.Radio(["Yes", "No"], label="Partner"),
            gr.Radio(["Yes", "No"], label="Dependents"),

            gr.Radio(["Yes", "No"], label="Phone Service"),
            gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines"),

            gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service"),

            gr.Radio(["Yes", "No", "No internet service"], label="Online Security"),
            gr.Radio(["Yes", "No", "No internet service"], label="Online Backup"),
            gr.Radio(["Yes", "No", "No internet service"], label="Device Protection"),
            gr.Radio(["Yes", "No", "No internet service"], label="Tech Support"),
            gr.Radio(["Yes", "No", "No internet service"], label="Streaming TV"),
            gr.Radio(["Yes", "No", "No internet service"], label="Streaming Movies"),

            gr.Radio(["Month-to-month", "One year", "Two year"], label="Contract"),
            gr.Radio(["Yes", "No"], label="Paperless Billing"),

            gr.Radio(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                label="Payment Method",
            ),
        ],
        outputs=[
            gr.Number(label="Churn Prediction (0 = No, 1 = Yes)"),
            gr.Number(label="Churn Probability"),
        ],
        title="Customer Churn Predictor",
        description="Predict customer churn using a trained ML model.",
    )
