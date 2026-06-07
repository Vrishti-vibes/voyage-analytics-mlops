# ✈️ Voyage Analytics

## AI Travel Intelligence Platform with MLOps

Voyage Analytics is a production-oriented Machine Learning and MLOps platform designed for travel intelligence. The platform provides flight price prediction, hotel recommendation, travel analytics, experiment tracking, model registry, API serving, and Dockerized deployment.

---

# Project Overview

The objective of this project is to demonstrate an end-to-end Machine Learning lifecycle including:

- Data preprocessing
- Feature engineering
- Model training
- Experiment tracking
- Model registry
- API deployment
- Interactive dashboard
- Containerization using Docker

---

# Features

## Flight Price Prediction

Predicts flight prices based on:

- Source city
- Destination city
- Flight type
- Distance
- Airline
- Month
- Day

---

## Hotel Recommendation System

Provides hotel recommendations based on:

- User preferences
- Ratings
- Price factors
- Travel requirements

---

## Travel Analytics Dashboard

Interactive analytics dashboard built using Streamlit.

Includes:

- Dataset insights
- KPIs
- Model outputs
- Business analytics

---

## MLflow Integration

Implemented:

- Experiment Tracking
- Metrics Logging
- Artifact Storage
- Model Registry
- Model Versioning

---

## FastAPI Backend

REST API endpoints available for:

- Flight prediction
- Hotel recommendation
- Gender prediction
- Health checks

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## Dockerized Deployment

Services deployed using Docker Compose:

- FastAPI Container
- Streamlit Container
- MLflow Container

---

# System Architecture

The architecture diagram is available in:

```
report/screenshots/06_architecture_diagram.png
```

---

# Tech Stack

## Programming Language

- Python 3.12

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy

## Backend

- FastAPI
- Uvicorn

## Frontend

- Streamlit

## MLOps

- MLflow

## Deployment

- Docker
- Docker Compose

---

# Project Structure

```text
MAJOR PROJECT
│
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── report/
│   └── screenshots/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_flight_model.py
│   ├── train_gender_model.py
│   ├── recommendation.py
│   └── predict.py
│
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Running the Project

## Clone Repository

```bash
git clone <repository-url>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Docker Deployment

```bash
docker compose up --build
```

---

# Available Services

## Streamlit Dashboard

```text
http://localhost:8501
```

---

## FastAPI

```text
http://localhost:8000
```

---

## FastAPI Swagger

```text
http://localhost:8000/docs
```

---

## MLflow

```text
http://localhost:5000
```

---

# Results

Successfully implemented:

- Flight Price Prediction Model
- Gender Classification Model
- Hotel Recommendation Engine
- MLflow Experiment Tracking
- Model Registry
- FastAPI Deployment
- Streamlit Dashboard
- Dockerized Deployment

---

# Future Scope

- Real-time flight data integration
- Cloud deployment on AWS
- Kubernetes orchestration
- CI/CD automation
- Recommendation engine enhancement
- Advanced forecasting models

---

# Conclusion

Voyage Analytics demonstrates a complete end-to-end Machine Learning and MLOps workflow, covering data preprocessing, model training, experiment tracking, model registry, API deployment, dashboard visualization, and Dockerized deployment.

The project showcases production-oriented Machine Learning practices suitable for academic major projects and MLOps portfolio demonstrations.