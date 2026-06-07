import os
import pandas as pd

os.makedirs("report", exist_ok=True)

hotels = pd.read_csv("data/processed/hotels_processed.csv")

def recommend_hotels(place, budget_per_day, stay_days, top_n=5):
    df = hotels.copy()

    df["place_match"] = df["place"].str.lower() == place.lower()
    df = df[df["place_match"]]

    if df.empty:
        return pd.DataFrame()

    df["estimated_total_cost"] = df["price"] * stay_days
    df["budget_gap"] = budget_per_day - df["price"]

    df = df[df["budget_gap"] >= 0]

    if df.empty:
        return pd.DataFrame()

    # Professional scoring system
    df["affordability_score"] = 1 - (df["price"] / budget_per_day)
    df["stay_efficiency_score"] = 1 / (1 + abs(df["days"] - stay_days))
    df["recommendation_score"] = (
        0.70 * df["affordability_score"] +
        0.30 * df["stay_efficiency_score"]
    )

    recommendations = (
        df[[
            "name",
            "place",
            "price",
            "days",
            "estimated_total_cost",
            "budget_gap",
            "recommendation_score"
        ]]
        .drop_duplicates(subset=["name", "place", "price"])
        .sort_values(by="recommendation_score", ascending=False)
        .head(top_n)
    )

    recommendations["recommendation_score"] = recommendations["recommendation_score"].round(4)
    recommendations["estimated_total_cost"] = recommendations["estimated_total_cost"].round(2)
    recommendations["budget_gap"] = recommendations["budget_gap"].round(2)

    return recommendations


if __name__ == "__main__":
    place = "Florianopolis (SC)"
    budget_per_day = 400
    stay_days = 3

    result = recommend_hotels(
        place=place,
        budget_per_day=budget_per_day,
        stay_days=stay_days,
        top_n=5
    )

    print("\nAI-Powered Hotel Recommendation Engine")
    print("=====================================")
    print(f"Destination       : {place}")
    print(f"Budget Per Day    : {budget_per_day}")
    print(f"Planned Stay Days : {stay_days}")

    if result.empty:
        print("\nNo suitable hotels found for the selected preferences.")
    else:
        print("\nTop Recommended Hotels:")
        print(result.to_string(index=False))

        result.to_csv("report/hotel_recommendations_sample.csv", index=False)
        print("\nRecommendation output saved at: report/hotel_recommendations_sample.csv")