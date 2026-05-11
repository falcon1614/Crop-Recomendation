from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pickle
from pathlib import Path
import uvicorn

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Model Paths
MODEL_PATH = BASE_DIR / "model_tuned.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Load Model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open(SCALER_PATH, "rb") as f:
    ms = pickle.load(f)

# Create FastAPI App
app = FastAPI(
    title="Crop Recommendation System"
)

# Mount Static Files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Templates
templates = Jinja2Templates(directory="templates")


# Home Route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# Prediction Route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Nitrogen: float = Form(...),
    Phosporus: float = Form(...),
    Potassium: float = Form(...),
    Temperature: float = Form(...),
    Humidity: float = Form(...),
    Ph: float = Form(...),
    Rainfall: float = Form(...)
):

    try:

        # Create Feature List
        feature_list = [
            Nitrogen,
            Phosporus,
            Potassium,
            Temperature,
            Humidity,
            Ph,
            Rainfall
        ]

        # Convert to NumPy Array
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale Input
        scaled_features = ms.transform(single_pred)

        # Predict
        prediction = model.predict(scaled_features)

        # Debug
        print("Prediction:", prediction)
        print("Type:", type(prediction[0]))

        # If model returns string directly
        crop = str(prediction[0])

        result = f"{crop} is the best crop to cultivate."

    except Exception as e:

        print("Error:", e)
        result = f"Error occurred: {e}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )


# Run Server
if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
