# Customer Churn Prediction API

A production-ready machine learning system for predicting customer churn, deployed as a cloud-native API on Google Cloud Run. This project demonstrates end-to-end ML engineering and MLOps best practices, from model training to scalable deployment.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)
[![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg)](https://www.terraform.io/)

## 🎯 Project Overview

This repository showcases a complete ML system that goes beyond model training to address real-world deployment challenges. The focus is on **production readiness, reliability, and maintainability** rather than just model accuracy.

### Key Features

- **Robust ML Pipeline** with schema validation and feature contracts
- **Experiment Tracking** using MLflow with Optuna hyperparameter optimization
- **RESTful API** built with FastAPI for programmatic access
- **Interactive UI** powered by Gradio for non-technical users
- **Infrastructure as Code** using Terraform for reproducible deployments
- **Cloud-Native Architecture** running on Google Cloud Run for scalability

## 🚀 Live Deployment

**Note:** The service runs on Google Cloud Platform free tier and may not always be available.

When active:
- **Interactive UI**: `[your-service-url]/ui`
- **API Documentation**: `[your-service-url]/docs`
- **Health Check**: `[your-service-url]/health`

## 📊 Machine Learning Approach

### Problem Statement
Binary classification to predict whether a customer will churn (leave the service).

### Dataset
**Telco Customer Churn** dataset from Kaggle containing:
- Customer demographics and account information
- Service subscriptions (phone, internet, streaming)
- Contract details and billing information
- Historical usage patterns

### Features
- **Tenure**: Length of customer relationship
- **Services**: Phone, internet, streaming subscriptions
- **Contract Type**: Month-to-month, one year, two year
- **Billing**: Payment method, monthly charges, total charges
- **Demographics**: Senior citizen status, partner, dependents

### Models Evaluated

| Model | Purpose | Key Characteristics |
|-------|---------|---------------------|
| Logistic Regression | Baseline | Simple, interpretable, fast inference |
| Random Forest | Ensemble | Handles non-linearity, feature importance |
| XGBoost | Production | Best performance, gradient boosting |

### Evaluation Metrics

- **ROC-AUC**: Overall model discrimination ability
- **Precision/Recall**: Trade-off between false positives and false negatives
- **Accuracy**: Overall correctness
- **Confusion Matrix**: Detailed error analysis

## 🏗️ Architecture

```
┌─────────────────┐
│   Data Layer    │  Telco Churn Dataset
└────────┬────────┘
         │
┌────────▼────────┐
│  Training       │  MLflow + Optuna
│  Pipeline       │  Model Validation
└────────┬────────┘
         │
┌────────▼────────┐
│  Model          │  Serialized Model
│  Artifacts      │  Feature Schema
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI        │  REST API + Gradio UI
│  Application    │  Pydantic Validation
└────────┬────────┘
         │
┌────────▼────────┐
│  Docker         │  Containerized Service
│  Container      │  Dependencies Locked
└────────┬────────┘
         │
┌────────▼────────┐
│  Cloud Run      │  Serverless Deployment
│  (Terraform)    │  Auto-scaling
└─────────────────┘
```

## 🛠️ Technology Stack

### ML & Data Science
- **scikit-learn**: Model training and preprocessing
- **XGBoost**: Gradient boosting implementation
- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **Pandas/NumPy**: Data manipulation

### API & Serving
- **FastAPI**: High-performance API framework
- **Pydantic**: Data validation and serialization
- **Gradio**: Interactive web UI
- **Uvicorn**: ASGI server

### Infrastructure & DevOps
- **Docker**: Containerization
- **Terraform**: Infrastructure as Code
- **Google Cloud Run**: Serverless container platform
- **Google Container Registry**: Docker image storage

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned and transformed data
├── notebooks/
│   └── exploration.ipynb     # EDA and experimentation
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Data cleaning
│   │   └── validation.py     # Schema validation
│   ├── models/
│   │   ├── train.py          # Training pipeline
│   │   └── evaluate.py       # Model evaluation
│   ├── api/
│   │   ├── main.py           # FastAPI application
│   │   ├── schemas.py        # Pydantic models
│   │   └── predict.py        # Inference logic
│   └── ui/
│       └── gradio_app.py     # Interactive interface
├── terraform/
│   ├── main.tf               # Cloud Run configuration
│   ├── variables.tf          # Input variables
│   └── outputs.tf            # Service endpoints
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- Docker
- Google Cloud SDK (for deployment)
- Terraform (for infrastructure)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python src/models/train.py
   ```

5. **Run API locally**
   ```bash
   uvicorn src.api.main:app --reload
   ```

6. **Access the application**
   - API: http://localhost:8000
   - Swagger docs: http://localhost:8000/docs
   - Gradio UI: http://localhost:8000/ui

### Docker Deployment

```bash
# Build image
docker build -t churn-prediction:latest .

# Run container
docker run -p 8000:8000 churn-prediction:latest
```

### Cloud Deployment

1. **Configure GCP credentials**
   ```bash
   gcloud auth application-default login
   ```

2. **Initialize Terraform**
   ```bash
   cd terraform
   terraform init
   ```

3. **Deploy infrastructure**
   ```bash
   terraform plan
   terraform apply
   ```

## 📝 API Usage

### Prediction Endpoint

**POST** `/predict`

```bash
curl -X POST "https://your-service-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.68,
  "prediction": "Yes",
  "confidence": "Medium",
  "model_version": "v1.2.0"
}
```

### Batch Prediction

**POST** `/predict/batch`

For processing multiple customers at once.

## 🔄 CI/CD Workflow

The deployment pipeline follows MLOps best practices:

```
git push (main)
    ↓
CI Pipeline (GitHub Actions)
    ↓
Docker Build & Push to GCR
    ↓
Terraform Apply
    ↓
Cloud Run Deployment
    ↓
Health Check & Validation
```

**Benefits:**
- Immutable deployments
- Reproducible infrastructure
- Automated testing
- Zero-downtime releases
- Rollback capability

## 🧪 Model Performance

| Model | ROC-AUC | Precision | Recall | Accuracy |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.68 | 0.71 | 0.79 |
| Random Forest | 0.86 | 0.73 | 0.75 | 0.82 |
| **XGBoost** | **0.89** | **0.77** | **0.79** | **0.85** |

*Note: Metrics shown are from validation set. See MLflow for detailed experiment tracking.*

## 🔐 Security Considerations

- Input validation using Pydantic schemas
- Rate limiting on API endpoints
- HTTPS enforcement in production
- No sensitive data logged
- Environment-based configuration

## 📈 Future Enhancements

- [ ] Model retraining pipeline with drift detection
- [ ] A/B testing framework for model comparison
- [ ] Feature store integration
- [ ] Monitoring dashboard (Prometheus + Grafana)
- [ ] Automated model versioning and registry
- [ ] Multi-cloud deployment support

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- FastAPI documentation and community
- MLflow and Optuna teams

## 📧 Contact

For questions or feedback, please open an issue or reach out at [your-email@example.com]

---

**Built with ❤️ focusing on production ML systems and MLOps best practices**


The API is deployed on Google Cloud Run. Due to free-tier limitations, the service may not always be running.

If available:

Interactive UI: /ui
API documentation (Swagger): /docs

