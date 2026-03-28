import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'customer_churn_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def generate_dummy_data(path, num_samples=1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(18.0, 120.0, size=num_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 10, size=num_samples)
    total_charges = np.clip(total_charges, 0, None) # Ensure no negative total charges
    
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.5, 0.3, 0.2])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples, p=[0.3, 0.5, 0.2])
    
    # Calculate churn probability roughly based on features so the model can capture patterns
    churn_prob = np.zeros(num_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0.0)
    churn_prob += np.where(tenure < 12, 0.3, -0.1)
    churn_prob += np.where(monthly_charges > 70, 0.2, 0.0)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'Churn': churn
    })
    df.to_csv(path, index=False)
    print(f"Generated synthetic customer data at {path}")

def train_model():
    if not os.path.exists(DATA_PATH):
        generate_dummy_data(DATA_PATH)
        
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Convert categorical variables to dummy variables
    df = pd.get_dummies(df, columns=['Contract', 'InternetService'], drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Save the expected feature names for API inference
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model pipeline...")
    # Using a Pipeline to couple the scaler and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print(f"Training Accuracy: {pipeline.score(X_train, y_train):.3f}")
    print(f"Testing Accuracy: {pipeline.score(X_test, y_test):.3f}")
    
    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the pipeline as churn_model.pkl
    model_path = os.path.join(MODEL_DIR, 'churn_model.pkl')
    joblib.dump(pipeline, model_path)
    
    # Also save individual scaler and features list to match project structure
    joblib.dump(pipeline.named_steps['scaler'], os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'features.pkl'))
    
    print(f"Successfully saved all model artifacts to {MODEL_DIR}")

if __name__ == "__main__":
    train_model()
