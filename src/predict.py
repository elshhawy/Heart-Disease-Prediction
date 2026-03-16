"""
src/predict.py
Inference script — load saved model and predict for new patients.
Run:   python src/predict.py
Import: from src.predict import predict_patient
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_names.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    model = joblib.load(model_path)
    with open(feature_path) as f:
        feature_names = json.load(f)
    return model, feature_names


def preprocess_patient(patient: dict, feature_names: list) -> pd.DataFrame:
    """Encode a raw patient dict into a model-ready DataFrame row."""
    df = pd.DataFrame([patient])

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    if "ExerciseAngina" in df.columns:
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    df = pd.get_dummies(df, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=True)

    # Align with training feature set
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


def predict_patient(patient: dict) -> dict:
    """
    Predict heart disease for a single patient dict.

    Returns:
        {
          "prediction":  0 or 1,
          "label":       "No Heart Disease" | "Heart Disease",
          "probability": float (0.0 – 1.0)
        }
    """
    model, feature_names = load_artifacts()
    X    = preprocess_patient(patient, feature_names)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {
        "prediction":  pred,
        "label":       "Heart Disease" if pred == 1 else "No Heart Disease",
        "probability": round(proba, 4),
    }


# ── Demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    demo_patients = [
        {   # High risk profile
            "Age": 63, "Sex": "M", "ChestPainType": "ASY",
            "RestingBP": 145, "Cholesterol": 233, "FastingBS": 1,
            "RestingECG": "Normal", "MaxHR": 150,
            "ExerciseAngina": "Y", "Oldpeak": 2.3, "ST_Slope": "Down"
        },
        {   # Low risk profile
            "Age": 35, "Sex": "F", "ChestPainType": "ATA",
            "RestingBP": 120, "Cholesterol": 180, "FastingBS": 0,
            "RestingECG": "Normal", "MaxHR": 172,
            "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up"
        },
    ]

    print("=" * 45)
    print("  Heart Disease Prediction — Demo")
    print("=" * 45)
    for i, p in enumerate(demo_patients, 1):
        result = predict_patient(p)
        print(f"\nPatient {i}  |  Age: {p['Age']} | Sex: {p['Sex']} | ChestPain: {p['ChestPainType']}")
        print(f"  → {result['label']}  (probability: {result['probability']:.1%})")
