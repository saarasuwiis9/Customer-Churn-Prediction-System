# Customer Churn Prediction API

A complete machine learning project featuring data synthesis, a scikit-learn training pipeline, a FastAPI prediction server, and a beautiful web interface.

## Project Structure

```text
Customer Churn Prediction API project
├── api/
│   └── app.py                  # FastAPI server providing the /predict endpoint
├── models/                     # Saved artifacts (auto-generated)
│   ├── churn_model.pkl
│   ├── features.pkl
│   └── scaler.pkl
├── dataset/                    # Training data (auto-generated)
│   └── customer_churn_data.csv
├── notebooks/
│   └── churn_training.ipynb    # Jupyter Notebook for exploratory analysis
├── src/
│   └── train.py                # Pipeline script for synthetic data & model training
├── web/
│   └── index.html              # Modern UI dashboard to interact with the API
├── project_paper.md            # Methodology & architectural document
├── requirements.txt            # Python dependencies
└── README.md                   # Setup instructions (this file)
```

## 🚀 Quick Start Guide

### 1. Install Dependencies
Open your terminal and install the required modules directly into your Python environment:

```bash
pip install -r requirements.txt
```

### 2. Generate Data and Train Model
Run the core training pipeline. This will auto-generate 1,000 synthetic customer records in `dataset/` and then train the RandomForest model, saving the configuration to `models/`.

```bash
python src/train.py
```

### 3. Start the Inference Server
Boot up the fast application using explicitly `uvicorn`:

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```
> Or alternatively, just run `python api/app.py`.

### 4. Open the Web App
Open the `web/index.html` file in any web browser. You'll see a premium-designed interface. Try submitting a customer record (e.g. Month-to-month contract, low tenure to see high churn risk).
