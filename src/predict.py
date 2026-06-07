import joblib
import pandas as pd

model = joblib.load("models/flight_price_model.pkl")

sample_input = pd.DataFrame([{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "time": 1.76,
    "distance": 676.53,
    "agency": "FlyingDrops",
    "month": 9,
    "day": 26
}])

prediction = model.predict(sample_input)

print("Sample Flight Price Prediction")
print("------------------------------")
print(f"Predicted Price: {prediction[0]:.2f}")