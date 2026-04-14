from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import uvicorn
import os

app = FastAPI(title="Customer Churn Prediction API", description="API for predicting telco customer churn", version="1.0.0")

def _load_background_data(scaler_obj, columns):
    data_path = "data/processed/clean_telco.csv"
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
    X = X.reindex(columns=columns, fill_value=0)
    X_scaled = pd.DataFrame(scaler_obj.transform(X), columns=columns)
    return X_scaled.sample(min(200, len(X_scaled)), random_state=42)

def _build_explainer(model_obj, background_df):
    model_name = model_obj.__class__.__name__.lower()
    if "xgb" in model_name or "forest" in model_name or "tree" in model_name:
        return shap.TreeExplainer(model_obj)
    if hasattr(model_obj, "coef_") and background_df is not None:
        return shap.LinearExplainer(model_obj, background_df)
    if background_df is not None:
        return shap.Explainer(model_obj.predict_proba, background_df)
    return None

# Load models and preprocessors eagerly
try:
    model_path = 'models/best_model.pkl' if os.path.exists('models/best_model.pkl') else 'models/xgboost_model.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    background_df = _load_background_data(scaler, feature_columns)
    explainer = _build_explainer(model, background_df)
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    explainer = None
    print(f"Failed to load model artifacts: {e}")

class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    
    # Convert input to DataFrame
    df = pd.DataFrame([customer.model_dump()])
    
    # One-hot encoding as done in training
    df_encoded = pd.get_dummies(df)
    
    # Align columns with training data
    df_aligned = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in df_encoded.columns:
            df_aligned[col] = df_encoded[col]
        else:
            df_aligned[col] = 0 # fill missing categorical features
            
    # Scale numerical features
    df_scaled = pd.DataFrame(scaler.transform(df_aligned), columns=feature_columns)
    
    # Predict
    churn_prob = model.predict_proba(df_scaled)[0, 1]
    prediction = int(churn_prob >= 0.5)
    
    top_features = []
    if explainer is not None:
        try:
            shap_exp = explainer(df_scaled)
            shap_vals = shap_exp.values
            sample_vals = shap_vals[0]
            # Binary classifiers can return per-class contributions with shape (n_features, 2).
            if isinstance(sample_vals, np.ndarray) and sample_vals.ndim == 2 and sample_vals.shape[1] == 2:
                sample_vals = sample_vals[:, 1]

            feature_importances = sorted(
                zip(feature_columns, sample_vals), key=lambda x: abs(x[1]), reverse=True
            )
            top_features = [{"feature": f, "contribution": float(v)} for f, v in feature_importances[:5]]
        except Exception:
            # Keep predictions available even if SHAP fails for a specific sample/model.
            top_features = []

    return {
        "churn_probability": float(churn_prob),
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "top_contributing_features": top_features
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
