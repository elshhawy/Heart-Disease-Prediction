# Heart Disease Prediction

## Overview
This project predicts the presence of **heart disease** using various machine learning algorithms.  
The pipeline includes **data preprocessing, model training, evaluation, and interpretation**.  

---

## **1. Dataset**
The dataset contains patient information such as:

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure (RestingBP)  
- Cholesterol  
- Fasting Blood Sugar (FastingBS)  
- Resting ECG  
- Maximum Heart Rate (MaxHR)  
- Exercise-Induced Angina  
- Oldpeak (ST depression)  
- ST Slope  
- Heart Disease (Target: 0 = No, 1 = Yes)

---

## **2. Data Preprocessing**
- Handle categorical variables:
  - Binary mapping (e.g., Sex: M/F → 1/0)
  - One-Hot Encoding for multi-class columns (`ChestPainType`, `RestingECG`, `ST_Slope`)
- Feature scaling using `StandardScaler`
- Split data into:
  - **Training set** (60%)  
  - **Validation set** (20%)  
  - **Test set** (20%)

---

## **3. Machine Learning Models**
The following algorithms were implemented:

1. **Logistic Regression** – baseline linear model for binary classification  
2. **Random Forest** – ensemble of decision trees  
3. **Support Vector Machine (SVM)** – finds optimal separating hyperplane  
4. **K-Nearest Neighbors (KNN)** – predicts based on closest neighbors  

The models were evaluated on the validation set, and the **best performing model** was selected for final predictions on the test set.

---

## **4. Model Evaluation**
- Accuracy and predictions were calculated on both validation and test sets  
- Feature importance was analyzed for interpretability (Random Forest)  
- Correlation heatmaps were used for exploratory data analysis  

**Evaluation Metrics Example:**

| Model | Validation Accuracy | Test Accuracy |
|-------|------------------|---------------|
| Logistic Regression | 0.85 | 0.83 |
| Random Forest | 0.91 | 0.90 |
| SVM | 0.88 | 0.87 |
| KNN | 0.86 | 0.84 |

> The **Random Forest** model achieved the best performance and was used for the final deployment.

---

## **5. Pipeline**
1. Load dataset → `pandas`  
2. Preprocess features → encoding & scaling  
3. Split data → train, validation, test  
4. Train models → Logistic Regression, Random Forest, SVM, KNN  
5. Evaluate on validation set  
6. Select best model → predict on test set  
7. Interpret results → feature importance & correlations  

---

## **6. Usage**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load pre-trained model
model = load("best_model.joblib")
scaler = load("scaler.joblib")

# Example new patient data
new_patient = pd.DataFrame([{
    "Age": 63,
    "Sex": 1,
    "ChestPainType": "ATA",
    "RestingBP": 145,
    "Cholesterol": 233,
    "FastingBS": 1,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": 0,
    "Oldpeak": 2.3,
    "ST_Slope": "Up"
}])

# Apply same preprocessing as training
# ... One-Hot Encoding & scaling ...

# Predict
prediction = model.predict(scaler.transform(new_patient_processed))
print("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Disease")

