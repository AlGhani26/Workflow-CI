import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Set MLflow tracking (local only)
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("personality-prediction")

# Load dataset
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Personality"])
y = df["Personality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Autologging
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Log parameters
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])

    # Save and log model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)

    print("✅ Model training selesai dan disimpan:", args.model_output)

    # Register model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "personality-classification"

    try:
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"✅ Model registered as '{registered_model_name}'")
    except Exception as e:
        print(f"❌ Failed to register model: {e}")

    # Serving instruction
    print(f"📍 To serve locally, run:\nmlflow models serve -m '{model_uri}' --port 5000")
