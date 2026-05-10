from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pickle
from pathlib import Path
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"

SCALER_PATH = BASE_DIR / "minmaxscaler.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))

ms = pickle.load(open(SCALER_PATH, "rb"))


# Create App
app = FastAPI(
    title="Crop Recommendation System"
)


# Static Files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Templates
templates = Jinja2Templates(directory="templates")


# Home Page
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

    # Feature Array
    feature_list = [
        Nitrogen,
        Phosporus,
        Potassium,
        Temperature,
        Humidity,
        Ph,
        Rainfall
    ]

    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale Features
    scaled_features = ms.transform(single_pred)

    # Prediction
    prediction = model.predict(scaled_features)

    # Crop Dictionary
    crop_dict = {
        1: "Rice",
        2: "Maize",
        3: "Jute",
        4: "Cotton",
        5: "Coconut",
        6: "Papaya",
        7: "Orange",
        8: "Apple",
        9: "Muskmelon",
        10: "Watermelon",
        11: "Grapes",
        12: "Mango",
        13: "Banana",
        14: "Pomegranate",
        15: "Lentil",
        16: "Blackgram",
        17: "Mungbean",
        18: "Mothbeans",
        19: "Pigeonpeas",
        20: "Kidneybeans",
        21: "Chickpea",
        22: "Coffee"
    }

    # Final Result
    if prediction[0] in crop_dict:

        crop = crop_dict[prediction[0]]

        result = f"{crop} is the best crop to cultivate."

    else:

        result = "Sorry, prediction could not be made."

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )


# Run Server
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
