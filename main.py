from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import joblib
from pydantic import BaseModel

class LandmarkRequest(BaseModel):
    landmarks: list[list[float]]

app = FastAPI(
    title="Maze Hand Gesture Control API",
    description="Predict maze action from 2D hand landmark coordinates",
    version="1.0.2"
)

Instrumentator().instrument(app).expose(app)

GESTURE_TO_ACTION_MAPPING = {
    16: "up",#"two up"
    2: "down",#"fist"
    3: "left",#"four"
    14: "right"#"three"
}

model = None
scaler = None

PREDICTIONS_COUNT = Counter("model_predictions_total", "Total number of predictions made")
GESTURE_COUNT = Counter("predicted_gesture_count", "Count of predicted gestures by class index", ['gesture'])

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    model = joblib.load("models/best_svc_model.pkl")
    scaler = joblib.load("models/MMscale.pkl")

@app.get("/")
def root():
    return {"message": "Maze Hand Gesture Control API is running."}

@app.post("/predict")
def predict_action(data: LandmarkRequest):
    if not data.landmarks or not all(len(pt) == 2 for pt in data.landmarks):
        raise HTTPException(status_code=400, detail="Each landmark must contain exactly two values (x, y).")

    flat_input = np.array(data.landmarks).flatten().reshape(1, -1)
    scaled_input = scaler.transform(flat_input)
    pred_class_index = model.predict(scaled_input)[0]
    action = GESTURE_TO_ACTION_MAPPING.get(pred_class_index, "unknown_action")

    PREDICTIONS_COUNT.inc()
    GESTURE_COUNT.labels(gesture=str(pred_class_index)).inc()

    return {"predicted_class_index": int(pred_class_index), "action": action}
