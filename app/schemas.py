from pydantic import BaseModel


class FlightPredictionInput(BaseModel):
    source: str
    destination: str
    flight_type: str
    time: float
    distance: float
    agency: str
    month: int
    day: int


class GenderPredictionInput(BaseModel):
    company: str
    age: int


class HotelRecommendationInput(BaseModel):
    place: str
    budget_per_day: float
    stay_days: int
    top_n: int = 5