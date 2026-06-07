import pandas as pd

# Load datasets
flights = pd.read_csv("data/raw/flights.csv")
hotels = pd.read_csv("data/raw/hotels.csv")
users = pd.read_csv("data/raw/users.csv")

# Convert date columns
flights["date"] = pd.to_datetime(flights["date"])
hotels["date"] = pd.to_datetime(hotels["date"])

# Date features
flights["year"] = flights["date"].dt.year
flights["month"] = flights["date"].dt.month
flights["day"] = flights["date"].dt.day

hotels["year"] = hotels["date"].dt.year
hotels["month"] = hotels["date"].dt.month
hotels["day"] = hotels["date"].dt.day

# Save processed files
flights.to_csv(
    "data/processed/flights_processed.csv",
    index=False
)

hotels.to_csv(
    "data/processed/hotels_processed.csv",
    index=False
)

users.to_csv(
    "data/processed/users_processed.csv",
    index=False
)

print("Data preprocessing completed")