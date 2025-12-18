Customer Churn Prediction API
A production-style machine learning system for customer churn prediction, deployed as a cloud-native API on Google Cloud Run using Docker and Terraform.
This project focuses on end-to-end ML engineering and MLOps practices, demonstrating how a trained model can be validated, packaged, deployed, and served reliably.

Project Goals

This repository demonstrates how to:
-  Build a robust ML pipeline with schema validation and feature contracts
-  Train, evaluate, and tune multiple models with MLflow + Optuna
-  Package a trained model into a FastAPI inference service
-  Provide both API access and an interactive UI (Gradio)
-  Deploy infrastructure using Infrastructure as Code (Terraform)
-  Run the system as a stateless, scalable Cloud Run service
The emphasis is on production readiness, not only model accuracy.

Machine Learning Approach

Problem: Binary customer churn prediction
Dataset: Telco Customer Churn dataset (Kaggle)
Target: Whether a customer will churn
Features: Customer tenure, Service subscriptions, Contract types, Billing and payment informations, etc

Models explored:
Logistic Regression (baseline)
Random Forest
XGBoost

Evaluation metrics:
ROC-AUC
Precision / Recall
Accuracy

Training, tuning, and evaluation are implemented in a shared, reusable pipeline, ensuring consistency between experimentation and production.
The deployment pipeline is intentionally decoupled from training, reflecting real-world ML system design.

Inference & Serving
-  FastAPI provides a REST API for predictions
-  Pydantic schemas enforce strict request validation
-  Gradio UI is mounted at /ui for interactive testing
  


The intended deployment workflow follows standard MLOps practices:

git push (main)
    ↓
CI pipeline
    - Build Docker image
    - Push to container registry
    - Terraform apply
    ↓
Cloud Run deploys a new revision

This separation ensures:
-  Immutable deployments
-  Reproducible infrastructure
-  Safe, repeatable releases
-  Live Deployment


The API is deployed on Google Cloud Run. Due to free-tier limitations, the service may not always be running.

If available:

Interactive UI: /ui
API documentation (Swagger): /docs

