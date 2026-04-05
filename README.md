# 🚀 Customer Churn Prediction System

## 📌 Overview

This project is a full-stack Machine Learning application that predicts whether a customer is likely to churn (leave a service) based on their behavior and service usage.

It demonstrates the complete ML pipeline — from data processing and model training to backend API development and frontend integration.

---

## 🧠 Problem Statement

Customer churn is a major challenge for businesses. Retaining customers is more cost-effective than acquiring new ones. This system helps identify customers at risk of leaving so companies can take preventive actions.

---

## ⚙️ Features

* Data preprocessing and feature engineering
* Machine learning model (Random Forest Classifier)
* Model pipeline with scaling and prediction
* FastAPI backend for real-time predictions
* Interactive web interface for user input
* Probability-based churn prediction

---

## 🏗️ Project Structure

```
project/
│
├── frontend/          # Web interface
├── backend/           # API and model logic
│   ├── app.py
│   ├── models/
│   └── src/
├── data/              # Dataset
├── notebooks/         # EDA and experiments
├── docs/              # Documentation
└── requirements.txt
```

---

## 🧪 Model Details

* Algorithm: Random Forest Classifier
* Features:

  * Tenure
  * Monthly Charges
  * Total Charges
  * Contract Type
  * Internet Service

The model predicts:

* `0` → Customer stays
* `1` → Customer churns

---

## 🔄 Workflow

1. Data is loaded and preprocessed
2. Categorical features are encoded
3. Model is trained using training data
4. Model and features are saved
5. FastAPI loads the model
6. User inputs data via web UI
7. API processes input and returns prediction

---

## ▶️ How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python backend/src/train.py
```

### 3. Run the API

```
uvicorn backend.app:app --reload
```

### 4. Open frontend

Open `frontend/index.html` in your browser

---

## 📊 API Endpoint

* **POST** `/predict`

### Sample Input:

```
{
  "tenure": 10,
  "MonthlyCharges": 70,
  "TotalCharges": 700,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic"
}
```

### Response:

```
{
  "prediction": 1,
  "churn_probability": 0.78,
  "customer_action": "High Risk of Churn",
  "confidence": "78%"
}
```

---

## ⚠️ Limitations

* Dataset is synthetic (not real-world data)
* Limited number of features
* Model evaluation uses only accuracy
* Not deployed online

---

## 🚀 Future Improvements

* Use real-world dataset
* Add more customer features
* Improve evaluation metrics (Precision, Recall, F1-score)
* Deploy backend and frontend
* Enhance UI/UX

---

## 💡 Key Learnings

* Building an ML model is only part of the process
* Integration (API + frontend) is critical
* Data quality strongly affects model performance

---

## 🔗 Author

**Sara Suwiis**

---


