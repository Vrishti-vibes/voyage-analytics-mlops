import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


DATA_PATH = "data/processed/users_processed.csv"
MODEL_PATH = "models/gender_model.pkl"
ENCODER_PATH = "models/gender_label_encoder.pkl"
METRICS_PATH = "report/gender_model_metrics.txt"

EXPERIMENT_NAME = "Voyage Analytics - Gender Prediction"
RUN_NAME = "RandomForestClassifier_Gender_Model"
REGISTERED_MODEL_NAME = "Voyage_Gender_Classification_Model"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 150
MAX_DEPTH = 10

os.makedirs("models", exist_ok=True)
os.makedirs("report", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

print("Loading processed users dataset...")
users = pd.read_csv(DATA_PATH)

features = ["company", "age"]
target = "gender"

X = users[features]
y = users[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_features = ["company"]
numeric_features = ["age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric_scaler", StandardScaler(), numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

with mlflow.start_run(run_name=RUN_NAME):

    print("Gender classification training started...")
    model.fit(X_train, y_train)
    print("Training completed.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print("\nGender Classification Model Results")
    print("-----------------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    with open(METRICS_PATH, "w") as f:
        f.write("Gender Classification Model Results\n")
        f.write("-----------------------------------\n")
        f.write("Model Type : RandomForestClassifier\n")
        f.write(f"Dataset    : {DATA_PATH}\n")
        f.write(f"Records    : {len(users)}\n")
        f.write(f"Features   : {', '.join(features)}\n")
        f.write(f"Target     : {target}\n\n")
        f.write(f"Accuracy   : {accuracy:.4f}\n")
        f.write(f"Precision  : {precision:.4f}\n")
        f.write(f"Recall     : {recall:.4f}\n")
        f.write(f"F1 Score   : {f1:.4f}\n\n")
        f.write("Classes:\n")
        f.write(", ".join(label_encoder.classes_))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report_text)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("dataset", DATA_PATH)
    mlflow.log_param("total_records", len(users))
    mlflow.log_param("target_column", target)
    mlflow.log_param("features", ", ".join(features))
    mlflow.log_param("classes", ", ".join(label_encoder.classes_))

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1)

    mlflow.log_artifact(METRICS_PATH)
    mlflow.log_artifact(ENCODER_PATH)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="gender_classification_model",
        registered_model_name="Voyage_Gender_Classification_Model"
    )

    print("\nModel saved at:", MODEL_PATH)
    print("Label encoder saved at:", ENCODER_PATH)
    print("Metrics saved at:", METRICS_PATH)
    print("MLflow registered model:", REGISTERED_MODEL_NAME)
    print("MLflow tracking completed successfully.")