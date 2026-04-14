import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def train_models():
    data_path = "data/processed/clean_telco.csv"
    if not os.path.exists(data_path):
        print("Processed data not found. Please run preprocessing.py first.")
        return

    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding for categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Save the order of columns for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(list(X_encoded.columns), 'models/feature_columns.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert scaled arrays back to dataframes to retain column names for SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
    X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_encoded.columns)
    
    joblib.dump(scaler, 'models/scaler.pkl')

    # Handling Class Imbalance via SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train)

    print(f"Original train shape: {X_train_scaled_df.shape}, Resampled train shape: {X_train_resampled.shape}")

    # Models definition
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    best_model_name = None
    best_model = None
    best_recall = 0
    model_metrics = {}

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled_df)
        y_prob = model.predict_proba(X_test_scaled_df)[:, 1] if hasattr(model, "predict_proba") else None
        
        print(f"\n--- {name} ---")
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        model_metrics[name] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if y_prob is not None else None
        }
        
        if recall > best_recall:
            best_recall = recall
            best_model_name = name
            best_model = model

    print(f"\nBest Model based on Recall: {best_model_name} (Recall: {best_recall:.4f})")

    # Save best model selected by recall
    joblib.dump(best_model, 'models/best_model.pkl')

    # Keep backward compatibility artifact name used in older API code
    if "XGBoost" in models:
        joblib.dump(models["XGBoost"], 'models/xgboost_model.pkl')

    # Persist metrics and selected model metadata
    os.makedirs('reports', exist_ok=True)
    with open('reports/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                "selection_metric": "recall",
                "best_model_name": best_model_name,
                "best_model_recall": float(best_recall),
                "models": model_metrics
            },
            f,
            indent=2
        )

    print("Models and preprocessors saved to models/ directory.")
    print("Model metrics saved to reports/model_metrics.json.")

if __name__ == "__main__":
    train_models()
