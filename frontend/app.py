import streamlit as st
import pandas as pd
import requests
import json
import google.generativeai as genai
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import CustomerData, predict as local_predict

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide", page_icon="📈")

st.title("📈 Customer Churn Prediction & Retention Dashboard")
st.markdown("Predict customer churn probability, understand key driving factors via SHAP, and generate customized retention strategies using Gemini GenAI.")

api_url = os.getenv("API_URL", "").strip()
if not api_url:
    try:
        api_url = st.secrets.get("API_URL", "").strip() if "API_URL" in st.secrets else ""
    except Exception:
        api_url = ""

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Dataset Insights"])

with tab1:
    st.header("Enter Customer Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    
    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    
    with col3:
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
    with col4:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges", value=29.85)
        total = st.number_input("Total Charges", value=29.85)
        
    if st.button("Predict Churn Risk"):
        customer_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total)
        }
        
        with st.spinner("Analyzing profile & explaining with SHAP..."):
            try:
                if api_url:
                    response = requests.post(api_url, json=customer_payload, timeout=30)
                    if response.status_code != 200:
                        st.error(f"Error from API: {response.text}")
                        st.stop()
                    data = response.json()
                else:
                    data = local_predict(CustomerData(**customer_payload))

                if data:
                    prob = data["churn_probability"] * 100
                    pred = data["prediction"]
                    features = data["top_contributing_features"]
                    
                    st.subheader(f"Prediction: {pred} (Probability: {prob:.2f}%)")
                    
                    if prob > 50:
                        st.error("High Risk of Churn!")
                    else:
                        st.success("Low Risk of Churn.")
                        
                    st.markdown("### 🔑 Key Drivers (SHAP values)")
                    feature_df = pd.DataFrame(features)
                    if not feature_df.empty:
                        st.bar_chart(feature_df.set_index("feature")["contribution"])
                    
                    if pred == "Churn" and api_key:
                        st.markdown("### 🤖 Generative AI Retention Strategy (Gemini)")
                        prompt = f"A customer is highly likely to churn. Key features driving this are: {features}. Profile: {customer_payload}. Suggest a personalized, professional 3-sentence email offering a retention deal or solution to keep them."
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        gen_res = model.generate_content(prompt)
                        st.info(gen_res.text)
                    elif not api_key:
                        st.warning("GEMINI_API_KEY environment variable not set. Exiting GenAI retention step.")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")

with tab2:
    st.header("Dataset Insights")
    figures_dir = ROOT_DIR / "reports" / "figures"
    figure_files = [
        "churn_distribution.png",
        "tenure_by_churn.png",
        "contract_vs_churn.png"
    ]

    shown = 0
    for fig in figure_files:
        fig_path = figures_dir / fig
        if os.path.exists(fig_path):
            st.image(fig_path, caption=fig.replace("_", " ").replace(".png", "").title(), use_container_width=True)
            shown += 1

    metrics_path = ROOT_DIR / "reports" / "model_metrics.json"
    if os.path.exists(metrics_path):
        st.markdown("### 🧪 Model Performance (Validation)")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        st.json(metrics_data)

    if shown == 0:
        st.info("No EDA figures found yet. Run `python src/features/preprocessing.py` first.")
