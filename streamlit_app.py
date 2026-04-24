import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Seasonal Disease Surge Prediction", layout="wide")

st.title("Seasonal Disease Surge Prediction")

# ---------- LOAD DATASET ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "disease_weather.csv")

st.write(f"Reading dataset from: {data_path}")

try:
    df = pd.read_csv(data_path)
    st.success("Dataset loaded successfully")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

# ---------- INPUTS ----------
st.sidebar.header("Input Parameters")

disease = st.sidebar.selectbox("DISEASE", ["Dengue", "Flu", "Malaria","typhoid","cholera","chickenpox","allergies"])
temp = st.sidebar.number_input("TEMPERATURE (^C)", value=00.0)
hum = st.sidebar.number_input("HUMIDITY (^C)", value=00.0)
rain = st.sidebar.number_input("Rainfall (^C)", value=00.0)

# ---------- PREDICT ----------
if st.sidebar.button("Predict"):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "TEMPERATURE (^C)": temp,
                "HUMIDITY (^C)": hum,
                "RAINFALL": rain,
                "DISEASE": disease
            },
            timeout=10
        )

        result = response.json()

        st.subheader("Prediction Result")
        st.success(f"Predicted Cases: {result['predicted_cases']}")
        st.warning(f"Risk Level: {result['risk_level']}")

        # ---------- GRAPH ----------
        st.subheader("Cases Trend + Prediction")

        if "cases" in df.columns:
            last_cases = df["CASES"].tail(10).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(last_cases, marker='o')
            ax.scatter(len(last_cases)-1, result['predicted_cases'])

            ax.set_xlabel("Recent Records")
            ax.set_ylabel("CASES")
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelrotation=45)

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Column 'cases' not found in dataset.")

    except Exception as e:
        st.error(f"Connection/API Error: {e}")