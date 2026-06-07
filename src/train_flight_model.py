import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/processed/flights_processed.csv"
MODEL_PATH = "models/flight_price_model.pkl"
METRICS_PATH = "report/flight_model_metrics.txt"

EXPERIMENT_NAME = "Voyage Analytics - Flight Price Prediction"
RUN_NAME = "RandomForestRegressor_Flight_Model"
REGISTERED_MODEL_NAME = "Voyage_Flight_Price_Model"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100
MAX_DEPTH = 18

os.makedirs("models", exist_ok=True)
os.makedirs("report", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

print("Loading processed flight dataset...")
df = pd.read_csv(DATA_PATH)

features = ["from", "to", "flightType", "time", "distance", "agency", "month", "day"]
target = "price"

X = df[features]
y = df[target]

categorical_features = ["from", "to", "flightType", "agency"]
numeric_features = ["time", "distance", "month", "day"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric_scaler", StandardScaler(), numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=MAX_DEPTH
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

with mlflow.start_run(run_name=RUN_NAME):

    print("Training started...")
    model.fit(X_train, y_train)
    print("Training completed.")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nFlight Price Prediction Model Results")
    print("------------------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        f.write("Flight Price Prediction Model Results\n")
        f.write("------------------------------------\n")
        f.write("Model Type : RandomForestRegressor\n")
        f.write(f"Dataset    : {DATA_PATH}\n")
        f.write(f"Records    : {len(df)}\n")
        f.write(f"Features   : {', '.join(features)}\n")
        f.write(f"Target     : {target}\n\n")
        f.write(f"MAE        : {mae:.2f}\n")
        f.write(f"RMSE       : {rmse:.2f}\n")
        f.write(f"R2 Score   : {r2:.4f}\n")

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("dataset", DATA_PATH)
    mlflow.log_param("total_records", len(df))
    mlflow.log_param("target_column", target)
    mlflow.log_param("features", ", ".join(features))

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    mlflow.log_artifact(METRICS_PATH)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="flight_price_model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    print("\nModel saved at:", MODEL_PATH)
    print("Metrics saved at:", METRICS_PATH)
    print("MLflow registered model:", REGISTERED_MODEL_NAME)
    print("MLflow tracking completed successfully.")