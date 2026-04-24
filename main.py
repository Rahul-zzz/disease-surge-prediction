from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Correct absolute path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)


class InputData(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    disease: str


@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array([[data.temperature, data.humidity, data.rainfall]])
        base_pred = model.predict(X)[0]

        factor = {
            "Dengue": 1.3,
            "Flu": 1.1,
            "Malaria": 1.5,
            "typhoid": 1.4,
            "cholera": 1.2,
            "chickenpox": 1.6,
            "allergies": 1.0
            
        }.get(data.disease, 1)

        final_pred = base_pred * factor

        if final_pred < 30:
            risk = "Low"
        elif final_pred < 60:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "predicted_cases": round(float(final_pred), 2),
            "risk_level": risk
        }

    except Exception as e:
        return {"error": str(e)}