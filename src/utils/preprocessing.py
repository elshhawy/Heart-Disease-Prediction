"""
utils/preprocessing.py
All data loading, validation, and feature engineering logic.
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and run basic sanity checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download heart.csv from:\n"
            "  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction\n"
            "and place it in the data/ folder."
        )
    df = pd.read_csv(path)

    # Sanity checks
    expected_cols = {
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
        "Oldpeak", "ST_Slope", "HeartDisease"
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns in dataset: {missing}"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values — handle them first."

    print(f"[✔] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"    Target balance → 0: {(df['HeartDisease']==0).sum()}  |  1: {(df['HeartDisease']==1).sum()}\n")
    return df


def preprocess(df: pd.DataFrame, cfg: dict, models_dir: str):
    """
    Encode categorical features and return X, y, feature_names.
    Saves feature_names.json for inference alignment.
    """
    df = df.copy()

    # Binary mapping from config
    for col, mapping in cfg["features"]["binary_map"].items():
        df[col] = df[col].map(mapping)

    # One-hot encoding
    df = pd.get_dummies(df, columns=cfg["features"]["onehot_cols"], drop_first=True)

    target = cfg["data"]["target"]
    X = df.drop(target, axis=1)
    y = df[target]

    feature_names = list(X.columns)
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    return X, y, feature_names


def split_data(X, y, cfg: dict):
    """
    Stratified 60 / 20 / 20 split with zero leakage.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    rs   = cfg["data"]["random_state"]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=rs, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=cfg["data"]["val_size"],
        random_state=rs, stratify=y_tv
    )
    print(f"[✔] Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test
