 Maze Hand Gesture Control API
A machine learning-based FastAPI application that predicts navigation actions (up, down, left, right) from hand landmarks for controlling a maze game. The model training and experimentation are tracked using MLflow, and the system is fully containerized with Docker, monitored using Prometheus and Grafana, and deployable via GitHub Actions and Railway.

Project Structure
bash
Copy
Edit
.
├── MLflow.py                  # SVC training with MLflow tracking
├── main.py                    # FastAPI application for inference
├── unit_test.py               # Unit test for prediction endpoint
├── hand_landmarks_data.csv    # Input dataset (not included here)
├── Dockerfile                 # Docker image configuration
├── docker-compose.yml         # Docker Compose for full stack (API + Monitoring)
├── prometheus.yml             # Prometheus scrape configuration
├── aws.yml                    # GitHub Actions CI/CD to Docker Hub & Railway
├── models/                    # Directory for trained models and scalers
├── requirements.txt           # Python dependencies
└── README.md                  # 📄 You are here
 Features
ML Model: Trained Support Vector Classifier (SVC) using best hyperparameters with 97% accuracy.

MLflow Tracking: Logs parameters, metrics, confusion matrix, and model artifacts.

FastAPI Inference: Predicts gestures in real-time from 2D hand landmark input.

Unit Testing: Simple test to validate the prediction endpoint.

Dockerized: Fully containerized backend for development and deployment.

Monitoring: Prometheus + Grafana integrated to monitor API metrics.

CI/CD Pipeline: GitHub Actions workflow to build and deploy via Railway.

Model Training (MLflow.py)

Dataset: hand_landmarks_data.csv (landmark positions & gesture labels)

Pipeline:

Drop z-coordinates

Encode labels with LabelEncoder

Scale features using MinMaxScaler

Train SVC(C=200, kernel='poly', gamma='scale')

Track metrics/artifacts with MLflow

Save:

Model: models/best_svc_model.pkl
 Accuracy : 0.97
Scaler: models/MMscale.pkl

Label encoder: models/label_encoder.pkl

⚙API Usage (main.py)
Start the API
bash
Copy
Edit
docker-compose up --build
# OR
uvicorn main:app --reload
Example Request
http
Copy
Edit
POST /predict
Content-Type: application/json

{
  "landmarks": [
    [x1, y1],
    [x2, y2],
    ...
    [xN, yN]
  ]
}
Response
json
Copy
Edit
{
  "predicted_class_index": 16,
  "action": "up"
}
Monitoring
Prometheus: collects metrics from /metrics endpoint

Grafana: visualize metrics like:

Total predictions

Predictions per class

Access via:

Prometheus: http://localhost:9090

Grafana: http://localhost:3001

Docker Instructions
Build & Run
bash
Copy
Edit
docker-compose up --build
Services
fastapi → app @ localhost:8000

prometheus → metrics @ localhost:9090

grafana → dashboards @ localhost:3001

cadvisor → container monitoring @ localhost:8080

☁️ CI/CD Pipeline (GitHub Actions)
Workflow: aws.yml

Trigger: Push/PR to main

Steps:

Install dependencies

Build & push Docker image to Docker Hub

(Optionally) deploy to Railway using Railway CLI

Set required secrets in your GitHub repository:

DOCKERHUB_USERNAME

DOCKERHUB_TOKEN

RAILWAY_TOKEN

Unit Test
Run test:

bash
Copy
Edit
python -m pytest unit_test.py
Tests include:

Valid landmark input returns 200 OK

Output includes class index and action label

Requirements
Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
