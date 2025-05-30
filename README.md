# Backend - MLOps Maze Game using Hand Gestures

This repository contains the backend API and infrastructure that powers the gesture-controlled maze game. It serves ML model predictions based on hand landmarks, handles input/output to the frontend, and ensures seamless communication between components.

## Hand Gesture to Maze Action Mapping

This section defines how the predicted hand gestures are translated into maze navigation commands.

| Predicted Gesture Label (from Model) | Original Hagrid Class Name | Maze Control Action |
| :----------------------------------- | :------------------------- | :------------------ |
| 16                                   | `like`                     | `up`                |
| 2                                    | `dislike`                  | `down`              |
| 3                                    | `four`                     | `right`             |
| 14                                   | `three`                    | `left`              |

*(Note: The API will return the string command, e.g., "up", "down","left" , "right" which the frontend will interpret.)*



Responsibilities

Serving hand gesture classification model
Mapping predicted gestures to maze control actions
Real-time inference with Mediapipe landmarks
Exposing API endpoints for frontend integration
Logging requests, predictions, and system health
Containerization and deployment setup
Tech Stack & Tools
FastAPI — REST API Framework
Used to expose model inference endpoints.



![API](https://github.com/user-attachments/assets/ec424a35-364e-48ce-8a96-c2cce3e74446)




MLflow — Experiment Tracking
Tracks training runs, hyperparameters, and model versions.



![Mlflow](https://github.com/user-attachments/assets/9387636c-e1bc-4be4-8316-cddb0b3ee861)




Docker — Containerization
Package the backend into reproducible containers.



![Docker_Desktop](https://github.com/user-attachments/assets/bcc01d94-c3cb-43b1-9fbb-bd2234fbc6e2)




Grafana + Prometheus — Monitoring
Used to monitor model latency, API throughput, and other metrics.



![grafana2](https://github.com/user-attachments/assets/3fffca7a-04aa-47f8-b124-34766ad63324)



## Monitoring Metrics
This API backend integrates Prometheus for real-time monitoring of various components using prometheus_fastapi_instrumentator. Below are the three selected metrics, categorized and explained:

1. Model-Related Metric
Metric: model_predictions_total

Reasoning:
This helps to Tracks the total number of predictions made by the ML model and monitor model usage frequency. A spike might indicate high load or production use, while a drop may suggest system or model failure. It’s also useful for performance benchmarking over time.



2. Data-Related Metric
Metric: predicted_gesture_count


Reasoning:
ThisTracks the distribution of predicted gesture classes and provides insight into model output balance. If one gesture is predicted disproportionately, it may reveal:
Data imbalance

Model overfitting

Incorrect mapping




3. Server-Related Metric
Metric: Auto-collected via prometheus_fastapi_instrumentator


Reasoning:
This helps Measures API latency and total number of requests to each endpoint and assess backend responsiveness and detect bottlenecks or outages. Sudden increases in latency or errors signal performance degradation.


Unit Tests — Model and API Testing
Ensure robustness of endpoints and logic with automated testing.




![API_Test](https://github.com/user-attachments/assets/383fe3d5-5ffa-4b6e-a0dc-b7160ba6983b)




API Endpoints
Method	Endpoint	Description
POST	/predict	Returns maze action from hand data
GET	/Root	Root endpoint with welcome message and API info
GET	/metrics	Prometheus-compatible metrics



Folder Structure

maze/
│
├── app/                                
│   ├── __init__.py               
│   ├── main.py                         
│   ├── Dockerfile                     
│   ├── requirements.txt               
├── models/                            
│   └── best_svc_model.pkl              
│   └── MMscale.pkl                     
├── mlartifacts/                        
├── mlruns/                             
├── main.py                             
├── MLflow.py                         
├── hand_landmarks_data.csv             
├── unit_test.py                      
├── docker-compose.yml                  
├── prometheus.yml                      
├── aws.yml                            
├── requirements.txt 
├── README.md                          


Deployment
Dockerhup

![Dockerhup](https://github.com/user-attachments/assets/eed170e5-aacd-4e6b-95dd-1409272752e0)




Railway_Deploy

![Railway_Deploy](https://github.com/user-attachments/assets/7cd1fd9b-abb9-480b-a72a-cd776497eebf)




Author
Eng. Khalid Ahmed Mohamed


