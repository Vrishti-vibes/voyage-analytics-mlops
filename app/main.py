import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import (
    FlightPredictionInput,
    GenderPredictionInput,
    HotelRecommendationInput
)

from src.recommendation import recommend_hotels

app = FastAPI(
    title="Voyage Analytics API",
    description="Production-ready MLOps API for travel intelligence, flight price prediction, hotel recommendation, and user classification.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

flight_model = joblib.load("models/flight_price_model.pkl")
gender_model = joblib.load("models/gender_model.pkl")
gender_label_encoder = joblib.load("models/gender_label_encoder.pkl")


@app.get("/")
def home():
    return {
        "project": "Voyage Analytics",
        "status": "API is running successfully",
        "modules": [
            "Flight Price Prediction",
            "Hotel Recommendation Engine",
            "User Gender Classification"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Voyage Analytics backend service is operational"
    }


@app.post("/predict-flight-price")
def predict_flight_price(data: FlightPredictionInput):
    input_df = pd.DataFrame([{
        "from": data.source,
        "to": data.destination,
        "flightType": data.flight_type,
        "time": data.time,
        "distance": data.distance,
        "agency": data.agency,
        "month": data.month,
        "day": data.day
    }])

    predicted_price = flight_model.predict(input_df)[0]

    return {
        "prediction_type": "Flight Price Prediction",
        "input": data.model_dump(),
        "predicted_price": round(float(predicted_price), 2),
        "currency": "BRL/USD dataset unit",
        "model": "Random Forest Regressor"
    }


@app.post("/predict-gender")
def predict_gender(data: GenderPredictionInput):
    input_df = pd.DataFrame([{
        "company": data.company,
        "age": data.age
    }])

    encoded_prediction = gender_model.predict(input_df)[0]
    predicted_gender = gender_label_encoder.inverse_transform([encoded_prediction])[0]

    return {
        "prediction_type": "User Gender Classification",
        "input": data.model_dump(),
        "predicted_gender": predicted_gender,
        "model": "Random Forest Classifier",
        "note": "This is an auxiliary model and depends on limited demographic features."
    }


@app.post("/recommend-hotels")
def hotel_recommendation(data: HotelRecommendationInput):
    recommendations = recommend_hotels(
        place=data.place,
        budget_per_day=data.budget_per_day,
        stay_days=data.stay_days,
        top_n=data.top_n
    )

    if recommendations.empty:
        return {
            "recommendation_type": "Hotel Recommendation",
            "message": "No suitable hotels found for selected preferences.",
            "recommendations": []
        }

    return {
        "recommendation_type": "Hotel Recommendation",
        "input": data.model_dump(),
        "recommendations": recommendations.to_dict(orient="records")
    }


@app.get("/model-info")
def model_info():
    return {
        "project_name": "Voyage Analytics",
        "mlops_components": [
            "Data Preprocessing",
            "Model Training",
            "Model Serialization",
            "API Deployment",
            "Recommendation Engine",
            "Monitoring-ready Endpoints"
        ],
        "models": {
            "flight_price_model": {
                "algorithm": "Random Forest Regressor",
                "artifact": "models/flight_price_model.pkl"
            },
            "gender_classification_model": {
                "algorithm": "Random Forest Classifier",
                "artifact": "models/gender_model.pkl"
            },
            "hotel_recommendation_engine": {
                "approach": "Weighted content-based recommendation scoring"
            }
        }
    }