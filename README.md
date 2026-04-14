# Customer Churn Prediction System

This is a production-ready system to predict customer churn, generate data insights via SHAP, and use Generative AI to recommend targeted retention strategies. This project utilizes the IBM Telco Customer Churn dataset.

## Business Problem
Retaining customers is often much more cost-effective than acquiring new ones. Churn prediction helps identify high-risk customers before they cancel, so the business can engage them.

## Solution Architecture
- **Data Engineering:** Pandas & NumPy for cleaning, handling class imbalance (SMOTE).
- **Modeling:** XGBoost, Random Forest, Logistic Regression. (Optimized for Recall to catch more potential churners with SHAP explainability overlay).
- **Backend API:** FastAPI exposing `/predict` endpoint that returns churn probability and SHAP top contributors.
- **Frontend Dashboard:** Streamlit UI allowing single profile analysis and GenAI retention emails via Google Gemini API.

## Repository Structure
```
├── app/
│   └── main.py              # FastAPI server
├── data/
│   ├── processed/           # Cleaned dataset (generated)
│   └── raw/                 # Original dataset
├── frontend/
│   └── app.py               # Streamlit application
├── models/                  # Saved .pkl models & scalars
├── notebooks/               # Jupyter Notebooks for EDA
├── reports/
│   └── figures/             # Auto-generated plots during EDA
├── src/
│   ├── features/            # Preprocessing scripts
│   └── models/              # Model training scripts
└── requirements.txt         # Project dependencies
```

## How to Run

1. **Environment Setup:**
   ```bash
   python -m venv venv
   # On Windows: .\venv\Scripts\Activate
   # On Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Processing & Training:**
   ```bash
   python src/features/preprocessing.py
   python src/models/train_model.py
   ```

3. **Start the API Backend:**
   In a new terminal:
   ```bash
   # Make sure your virtual environment is activated
   uvicorn app.main:app --reload
   ```

4. **Start the Streamlit Frontend Dashboard:**
   In another terminal:
   ```bash
   # Make sure your virtual environment is activated
   # Set the Gemini API key beforehand
   export GEMINI_API_KEY="your-api-key"   # For Windows: set GEMINI_API_KEY="your-api-key"
   streamlit run frontend/app.py
   ```

## Demo & Explainability
- Go to `http://localhost:8501/` to use the UI.
- Enter customer characteristics and click "Predict Churn Risk".
- The system will fetch the SHAP values from the XGBoost prediction and explicitly state what variables are driving the score.
- The GenAI agent uses these SHAP values to dynamically craft a personalized retention email.
