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



📌 Responsibilities
Serving hand gesture classification model
Mapping predicted gestures to maze control actions
Real-time inference with Mediapipe landmarks
Exposing API endpoints for frontend integration
Logging requests, predictions, and system health
Containerization and deployment setup
Tech Stack & Tools
FastAPI — REST API Framework
Used to expose model inference endpoints.


![API](https://github.com/user-attachments/assets/d86c49e6-6916-485d-9d4d-9576d66c2860)



MLflow — Experiment Tracking
Tracks training runs, hyperparameters, and model versions.

![Mlflow](https://github.com/user-attachments/assets/0f73954f-9f4f-44b5-90a7-3e3f1ff95480)




Docker — Containerization
Package the backend into reproducible containers.


![Docker_Desktop](https://github.com/user-attachments/assets/06084005-5ece-4039-b119-42119f820ce6)




Grafana + Prometheus — Monitoring
Used to monitor model latency, API throughput, and other metrics.


![grafana2](https://github.com/user-attachments/assets/1ed100e1-8867-4ed2-ae2c-cc311e8dfcc9)




Unit Tests — Model and API Testing
Ensure robustness of endpoints and logic with automated testing.


![API_Test](https://github.com/user-attachments/assets/b7d36399-47c7-41ad-b4f0-da463cf0884a)





API Endpoints
Method	Endpoint	Description
POST	/predict	Returns maze action from hand data
GET	/Root	Root endpoint with welcome message and API info
GET	/metrics	Prometheus-compatible metrics



Folder Structure
graphql
Copy
Edit
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



![Dockerhup](https://github.com/user-attachments/assets/5bac244a-d564-4c77-8f7a-64144ebaa045)

![Railway_Deploy](https://github.com/user-attachments/assets/d9f74f34-5f56-4b0f-934a-dd36fad8267b)






Author
Eng. Khalid Ahmed Mohamed


