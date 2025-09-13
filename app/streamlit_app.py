import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.write("Upload data or enter a single customer's details to predict churn.")

model_path = Path("models/best_model.joblib")
info_path = Path("artifacts/feature_info.json")

if not model_path.exists() or not info_path.exists():
    st.warning("Model artifacts not found. Please run training first: `python src/train.py --data data/Telco-Customer-Churn.csv`")
    st.stop()

model = joblib.load(model_path)
with open(info_path, "r") as f:
    feature_info = json.load(f)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Single Customer")
    inputs = {}
    # Build dynamic form from feature schema
    for col in feature_info.get("numeric", []):
        inputs[col] = st.number_input(col, value=0.0)
    for col in feature_info.get("categorical", []):
        # Free text input keeps it generic across datasets
        inputs[col] = st.text_input(col, value="")

    if st.button("Predict Churn"):
        X = pd.DataFrame([inputs])
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)
        st.metric("Churn Probability", f"{prob:.2%}")
        st.write("Prediction:", "Churn" if pred == 1 else "No Churn")

with tab2:
    st.subheader("Batch CSV Upload")
    st.write("Upload a CSV with the same columns as your training data (except the target).")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = df.copy()
        out["churn_probability"] = probs
        out["prediction"] = preds
        st.dataframe(out.head(20))
        st.download_button("Download Predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
