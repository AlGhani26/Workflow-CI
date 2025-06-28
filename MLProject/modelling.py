import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Set MLflow tracking
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("personality-prediction")

# Load dataset
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Personality"])
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Autologging optional
mlflow.sklearn.autolog(log_models=False)

# with mlflow.start_run():
model = RandomForestClassifier(random_state=args.random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")

# Manual logging
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_param("train_size", X_train.shape[0])
mlflow.log_param("test_size", X_test.shape[0])
mlflow.log_param("features", X_train.shape[1])

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = "diagram/confusion_matrix.png"
plt.savefig(cm_path)
mlflow.log_artifact(cm_path)
plt.close()

# Save model
os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
joblib.dump(model, args.model_output)
mlflow.log_artifact(args.model_output)

print(f"✅ Model trained and saved to: {args.model_output}")

# Register to local model registry
run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/model"
registered_model_name = "personality-prediction-model"

try:
    mlflow.sklearn.log_model(model, "model")
    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"✅ Model registered as '{registered_model_name}'")
except Exception as e:
    print(f"❌ Failed to register model: {e}")

print(f"📍 To serve the model, use:\nmlflow models serve -m '{model_uri}' --port 5000")
