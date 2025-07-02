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

# =======================
# CLI Argument Parsing
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# =======================
# MLflow Setup
# =======================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("personality-prediction")
mlflow.sklearn.autolog(log_models=False)

# =======================
# Load Dataset
# =======================
df = pd.read_csv(args.data_path)

if "Personality" not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'Personality'.")

X = df.drop(columns=["Personality"])
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# =======================
# Start MLflow run
# =======================
mlflow.start_run()
run_id = mlflow.active_run().info.run_id

model = RandomForestClassifier(random_state=args.random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

# Logging
mlflow.log_param("data_path", args.data_path)
mlflow.log_param("train_size", X_train.shape[0])
mlflow.log_param("test_size", X_test.shape[0])
mlflow.log_param("features_count", X_train.shape[1])

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)

# Save model
os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
joblib.dump(model, args.model_output)
mlflow.log_artifact(local_path=args.model_output, artifact_path="models")

print("✅ Model disimpan ke:", args.model_output)

# Register model
try:
    mlflow.sklearn.log_model(model, "model")
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="personality-classification")
    print("✅ Model registered as 'personality-classification'")
except Exception as e:
    print(f"❌ Model registration failed: {e}")

# Serve instruction
print(f"📍 Serve model locally with:\nmlflow models serve -m 'runs:/{run_id}/model' --port 5000")

# End MLflow run
mlflow.end_run()
