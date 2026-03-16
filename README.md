# 🫀 Heart Disease Prediction

> A production-ready machine learning pipeline for predicting heart disease from clinical patient data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Tests](https://img.shields.io/badge/tests-pytest-green?logo=pytest)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## 📌 Overview

This project builds a **binary classification pipeline** to predict whether a patient has heart disease based on 11 clinical features. The pipeline covers the full ML lifecycle: data validation, preprocessing, feature engineering, cross-validated model training, evaluation, visualization, and serialized inference.

**What makes this pipeline production-aware:**
- Zero data leakage — `StandardScaler` is fitted inside `sklearn.Pipeline`, never on validation/test data
- 5-fold stratified cross-validation for reliable generalization estimates
- Best model selected by validation AUC, evaluated once on a held-out test set
- Config-driven hyperparameters via `configs/config.yaml` — no hardcoded values in source
- Unit-tested preprocessing and inference with `pytest`
- Clean module separation: `utils/preprocessing`, `utils/evaluation`, `utils/visualization`

---

## 📊 Results

| Model | CV AUC (5-fold) | Val Accuracy | Val AUC |
|-------|----------------|-------------|---------|
| Logistic Regression | ~0.930 ± 0.023 | ~86% | ~0.932 |
| Gradient Boosting | ~0.920 ± 0.030 | ~82% | ~0.920 |
| SVM (RBF) | ~0.915 ± 0.039 | ~82% | ~0.915 |
| Random Forest | ~0.900 ± 0.033 | ~80% | ~0.900 |
| KNN | ~0.851 ± 0.040 | ~76% | ~0.860 |

> Exact values appear in `outputs/results_summary.json` after running `train.py`.

---

## 🗂️ Project Structure

```
heart_disease_project/
│
├── 📂 data/
│   └── heart.csv                        ← place dataset here (see below)
│
├── 📂 src/
│   ├── __init__.py
│   ├── train.py                         ← main training entry point
│   ├── predict.py                       ← inference for new patients
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py             ← load, encode, split
│       ├── evaluation.py                ← CV, metrics, summary
│       └── visualization.py            ← all plots
│
├── 📂 configs/
│   └── config.yaml                      ← hyperparameters & paths
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb                     ← exploratory data analysis
│   ├── 02_feature_engineering.ipynb     ← feature importance & new features
│   └── 03_model_comparison.ipynb        ← detailed model comparison
│
├── 📂 tests/
│   ├── test_preprocessing.py            ← unit tests: encoding, splits
│   ├── test_predict.py                  ← unit tests: inference pipeline
│   └── test_pipeline.py                 ← integration: full run
│
├── 📂 models/                           ← auto-generated after training
│   ├── best_model.pkl
│   └── feature_names.json
│
├── 📂 plots/                            ← auto-generated after training
│   ├── eda.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── correlation_heatmap.png
│
├── 📂 outputs/                          ← auto-generated after training
│   └── results_summary.json
│
├── 📂 docs/
│   ├── PROJECT_OVERVIEW.md              ← architecture & design decisions
│   └── DATA_DICTIONARY.md              ← feature descriptions & encoding
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 📥 Dataset Setup

**Download the dataset from Kaggle:**

1. Go to: [kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
2. Click **Download** → extract the ZIP
3. Place `heart.csv` in the `data/` folder:

```
heart_disease_project/
└── data/
    └── heart.csv   ✅
```

The file must be named exactly `heart.csv`. The dataset is free and requires a Kaggle account.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as a package
pip install -e .
```

---

## 🚀 Usage

### Train
```bash
python src/train.py
```

Output:
```
═══════════════════════════════════════════════════════
  Heart Disease Prediction — Training Pipeline
═══════════════════════════════════════════════════════

[✔] Loaded: 918 rows × 12 columns
[✔] Split → Train: 550 | Val: 184 | Test: 184

── Cross-Validation (5-fold) ──
  [Logistic Regression]  AUC: 0.9300 ± 0.0233
  [Random Forest]        AUC: 0.8992 ± 0.0332
  ...

── Final Test Evaluation ──
  ★  Best model : Logistic Regression
     Accuracy   : 0.9130
     AUC        : 0.9700

[✔] Model saved → models/best_model.pkl
[✔] Saved: plots/roc_curves.png
[✔] Saved: outputs/results_summary.json
```

### Predict for a new patient
```bash
python src/predict.py
```

Or import in your own code:
```python
from src.predict import predict_patient

patient = {
    "Age": 58, "Sex": "M", "ChestPainType": "ASY",
    "RestingBP": 140, "Cholesterol": 250, "FastingBS": 1,
    "RestingECG": "Normal", "MaxHR": 130,
    "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"
}

result = predict_patient(patient)
# → {'prediction': 1, 'label': 'Heart Disease', 'probability': 0.892}
```

### Run tests
```bash
pytest tests/ -v
```

### Explore notebooks
```bash
jupyter notebook notebooks/
```

---

## 🧠 Technical Notes

### No Data Leakage
All models use `sklearn.Pipeline` — the `StandardScaler` is fitted **only on training data** and applied to val/test using those statistics:

```python
Pipeline([
    ("scaler", StandardScaler()),   # fit on train only
    ("clf",    LogisticRegression())
])
```

### Evaluation Hierarchy

| Set | Size | Purpose |
|-----|------|---------|
| Train | 60% | Fit model weights |
| Validation | 20% | Compare models, select best |
| Test | 20% | Single final evaluation — reported once |

### Why AUC over Accuracy?
AUC is threshold-independent and more robust for medical classification tasks where class imbalance or decision thresholds matter.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥ 1.3 | Modeling, pipelines, CV |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib | ≥ 3.7 | Visualization |
| seaborn | ≥ 0.12 | Statistical plots |
| joblib | ≥ 1.3 | Model serialization |
| PyYAML | ≥ 6.0 | Config management |
| pytest | ≥ 7.4 | Unit & integration tests |

---

## 📄 License

MIT License — free to use, modify, and distribute.
