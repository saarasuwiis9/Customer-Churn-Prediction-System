from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Enable CORS for the web frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts safely
model_pipeline = None
expected_features = None

def load_models():
    global model_pipeline, expected_features
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        expected_features = joblib.load(FEATURES_PATH)
        print("Models successfully loaded into memory.")
    else:
        print("Warning: Model artifacts not found. Please run src/train.py first.")

load_models()

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str

@app.get("/")
def read_root():
    return {"status": "Customer Churn API is running."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    global model_pipeline, expected_features
    if model_pipeline is None or not expected_features:
        # Try loading again just in case model was trained after startup
        load_models()
        if model_pipeline is None or not expected_features:
            raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
        
    try:
        # Convert input to DataFrame
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        df = pd.DataFrame([input_dict])
        
        # Explicit dummy encoding for single-row inference
        contract = df['Contract'].iloc[0]
        internet = df['InternetService'].iloc[0]
        
        # Drop the original categorical columns
        df = df.drop(['Contract', 'InternetService'], axis=1)
        
        # Add all expected columns with default 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
                
        # Set the corresponding dummy variables to 1
        contract_col = f"Contract_{contract}"
        if contract_col in df.columns:
            df[contract_col] = 1
            
        internet_col = f"InternetService_{internet}"
        if internet_col in df.columns:
            df[internet_col] = 1
                
        # Filter and order columns properly
        X = df[expected_features]
        
        # Predict using the loaded pipeline (which automatically scales the data)
        probability = float(model_pipeline.predict_proba(X)[0][1])  # type: ignore
        prediction = int(model_pipeline.predict(X)[0])  # type: ignore
        
        # Format strings instead of round() to satisfy strict IDE type checkers
        churn_prob_rounded = float(f"{probability:.3f}")
        
        if prediction == 1:
            conf_str = f"{probability * 100:.1f}%"
        else:
            conf_str = f"{(1.0 - probability) * 100:.1f}%"
        
        return {
            "prediction": prediction,
            "churn_probability": churn_prob_rounded,
            "customer_action": "High Risk of Churn" if prediction == 1 else "Low Risk",
            "confidence": conf_str
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
