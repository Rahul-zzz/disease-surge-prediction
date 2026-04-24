import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv("data/disease_weather.csv")

X = df[["temperature", "humidity", "rainfall"]]
y = df["cases"]

model = LinearRegression()
model.fit(X, y)

# Save model inside api folder
os.makedirs("api", exist_ok=True)
joblib.dump(model, "api/model.pkl")

print("Model saved inside api folder!")