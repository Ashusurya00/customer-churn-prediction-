import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    print("Cleaning data...")
    # Replace empty spaces with NaN in TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    # Convert TotalCharges to float
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    
    # Drop rows with NaN
    df.dropna(inplace=True)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    return df

def perform_eda(df, output_dir):
    print("Performing EDA...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Tenure by Churn
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True)
    plt.title('Tenure Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'tenure_by_churn.png'))
    plt.close()
    
    # 3. Contract Type by Churn
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Contract Type vs Churn')
    plt.savefig(os.path.join(output_dir, 'contract_vs_churn.png'))
    plt.close()

def main():
    raw_data_path = "data/raw/Telco-Customer-Churn.csv"
    processed_data_path = "data/processed/clean_telco.csv"
    reports_dir = "reports/figures"
    
    df = load_data(raw_data_path)
    df = clean_data(df)
    perform_eda(df, reports_dir)
    
    print(f"Saving cleaned dataset to {processed_data_path}...")
    df.to_csv(processed_data_path, index=False)
    print("EDA and Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
