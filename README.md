# ✈️ Voyage Analytics

## AI-Powered Travel Intelligence Platform with End-to-End MLOps

Voyage Analytics is a production-oriented Machine Learning and MLOps platform developed to demonstrate the complete lifecycle of modern AI applications. The platform combines predictive analytics, recommendation systems, experiment tracking, model management, API serving, interactive visualization, and containerized deployment within a unified ecosystem.

Designed as an end-to-end solution, the project showcases industry-standard MLOps practices including model versioning, experiment reproducibility, deployment automation, and scalable service orchestration.

---

## Project Highlights

* Flight Price Prediction using Machine Learning
* Hotel Recommendation Engine
* Travel Analytics Dashboard
* FastAPI-Based Model Serving
* MLflow Experiment Tracking
* Model Registry & Version Management
* Dockerized Multi-Service Deployment
* Streamlit Interactive Dashboard
* Production-Oriented MLOps Workflow

---

## Business Problem

Travel pricing and accommodation decisions are influenced by multiple dynamic factors including destination, airline, travel dates, distance, and customer preferences. Traditional static systems often fail to provide personalized insights and predictive capabilities.

Voyage Analytics addresses this challenge by integrating predictive modeling, recommendation systems, and real-time analytics into a centralized travel intelligence platform.

---

## Core Modules

### Flight Price Prediction

A supervised Machine Learning model developed to estimate airfare based on travel attributes such as:

* Origin City
* Destination City
* Airline
* Flight Type
* Travel Distance
* Travel Month
* Travel Day

The model enables data-driven travel planning and pricing insights.

---

### Hotel Recommendation Engine

A recommendation system designed to assist travelers in identifying suitable accommodations based on:

* User Preferences
* Hotel Ratings
* Pricing Factors
* Travel Requirements
* Personalized Recommendation Logic

---

### Travel Analytics Dashboard

An interactive Streamlit dashboard provides:

* Travel Data Insights
* KPI Monitoring
* Prediction Outputs
* Recommendation Results
* Business Intelligence Visualizations
* Operational Analytics

---

### FastAPI Inference Layer

Production-ready REST APIs have been implemented using FastAPI for real-time model inference.

Available endpoints include:

* Flight Price Prediction
* Hotel Recommendation
* Gender Classification
* Health Monitoring

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

### MLflow MLOps Integration

The platform incorporates MLflow to manage the Machine Learning lifecycle.

Implemented Components:

* Experiment Tracking
* Parameter Logging
* Metric Monitoring
* Artifact Management
* Model Registry
* Model Version Control

This ensures reproducibility and governance throughout the model lifecycle.

---

### Dockerized Deployment

The complete platform is containerized using Docker and orchestrated through Docker Compose.

Services include:

* FastAPI Backend Container
* Streamlit Dashboard Container
* MLflow Tracking Server Container

Benefits:

* Reproducible Environments
* Simplified Deployment
* Environment Isolation
* Scalability Readiness

---

## System Architecture

Architecture Diagram:

```text
report/screenshots/06_architecture_diagram.png
```

The architecture follows a modular MLOps design pattern consisting of:

1. Data Layer
2. Data Processing Layer
3. Machine Learning Layer
4. Model Management Layer
5. API Serving Layer
6. Visualization Layer
7. Containerization Layer

---

## Technology Stack

### Programming Language

* Python 3.12

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-Learn

### Backend Development

* FastAPI
* Uvicorn

### Frontend & Visualization

* Streamlit

### MLOps

* MLflow

### Containerization

* Docker
* Docker Compose

### Version Control

* Git
* GitHub

---

## Project Structure

```text
voyage-analytics-mlops
│
├── app/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
│
├── k8s/
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
├── README.md
└── .gitignore
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/Vrishti-vibes/voyage-analytics-mlops.git
cd voyage-analytics-mlops
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Using Docker

```bash
docker compose up --build
```

---

## Available Services

### Streamlit Dashboard

```text
http://localhost:8501
```

### FastAPI Application

```text
http://localhost:8000
```

### FastAPI Swagger UI

```text
http://localhost:8000/docs
```

### MLflow Tracking Server

```text
http://localhost:5000
```

---

## Achievements

Successfully implemented:

* End-to-End Machine Learning Pipeline
* Flight Price Prediction Model
* Gender Classification Model
* Hotel Recommendation Engine
* MLflow Experiment Tracking
* Model Registry Integration
* FastAPI Deployment
* Streamlit Analytics Dashboard
* Dockerized Multi-Service Architecture
* Git-Based Version Control Workflow

---

## Future Enhancements

* Real-Time Flight Data Integration
* Cloud Deployment on AWS
* Kubernetes Orchestration
* CI/CD Pipeline Automation
* Advanced Recommendation Algorithms
* Automated Model Retraining
* Monitoring & Alerting Framework
* Distributed MLOps Infrastructure

---

## Conclusion

Voyage Analytics demonstrates a complete production-oriented Machine Learning and MLOps ecosystem that integrates data processing, predictive modeling, experiment management, API deployment, interactive analytics, and containerized infrastructure.

The project reflects modern software engineering and MLOps practices while serving as a scalable foundation for intelligent travel analytics applications.

---

## Author

**Kumari Vrishti**

B.Tech Computer Science Engineering

Machine Learning | MLOps | Backend Development | Data Analytics
