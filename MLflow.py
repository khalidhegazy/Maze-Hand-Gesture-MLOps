import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset and preprocess
data = pd.read_csv("hand_landmarks_data.csv")
data = data.drop(columns=[col for col in data.columns if 'z' in col])  # Drop z-coordinates

# Encode labels
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Split data into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.1, random_state=42, stratify=labels
)

# Apply MinMaxScaler
MM_scaler = MinMaxScaler()
features_train = MM_scaler.fit_transform(features_train)
features_test = MM_scaler.transform(features_test)

# Save scaler and label encoder locally
scaler_filename = "models/MMscale.pkl"
label_encoder_filename = "models/label_encoder.pkl"
joblib.dump(MM_scaler, scaler_filename)
joblib.dump(label_encoder, label_encoder_filename)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("HandGesture_Best_SVC_Training")

# Best hyperparameters found
best_C = 200
best_gamma = 'scale'
best_kernel = 'poly'

with mlflow.start_run(run_name="Best_SVC_Model_Run") as run:
    # Log hyperparameters
    mlflow.log_param("SVC_C", best_C)
    mlflow.log_param("SVC_gamma", best_gamma)
    mlflow.log_param("SVC_kernel", best_kernel)

    # Train SVC model
    svm_model = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
    svm_model.fit(features_train, labels_train)

    # Save the trained SVC model locally
    joblib.dump(svm_model, "models/best_svc_model.pkl")

    # Predict on train and test data
    train_predictions = svm_model.predict(features_train)
    test_predictions = svm_model.predict(features_test)

    # Calculate metrics
    train_accuracy = accuracy_score(labels_train, train_predictions)
    test_accuracy = accuracy_score(labels_test, test_predictions)
    test_precision = precision_score(labels_test, test_predictions, average='weighted', zero_division=0)
    test_recall = recall_score(labels_test, test_predictions, average='weighted', zero_division=0)
    test_f1 = f1_score(labels_test, test_predictions, average='weighted', zero_division=0)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate confusion matrix plot
    cm = confusion_matrix(labels_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Best SVC Model")
    plt.tight_layout()
    cm_path = "confusion_matrix_best_svc.png"
    plt.savefig(cm_path)
    plt.close()  # Prevent display issues

    # Log confusion matrix plot as artifact
    mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
    os.remove(cm_path)  # Clean up local file

    # Log the trained SVC model with MLflow
    mlflow.sklearn.log_model(svm_model, artifact_path="best_svm_model_artifact")

    # Log the scaler and label encoder artifacts
    mlflow.log_artifact(scaler_filename, artifact_path="preprocessing_scalers")
    mlflow.log_artifact(label_encoder_filename, artifact_path="preprocessing_scalers")

    # Print MLflow run ID for reference
    print(f"MLflow Run ID: {run.info.run_id}")
