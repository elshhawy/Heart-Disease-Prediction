# Project Overview — Heart Disease Prediction

## Objective
Build a binary classification model to predict the presence of heart disease (1 = Disease, 0 = No Disease) from 11 clinical features, using a clean, production-aware ML pipeline.

---

## Pipeline Architecture

```
heart.csv
    │
    ▼
load_data()          → validate shape, nulls, expected columns
    │
    ▼
preprocess()         → binary map + one-hot encode + save feature_names.json
    │
    ▼
split_data()         → stratified 60 / 20 / 20  (no leakage)
    │
    ├──► X_train, y_train
    │         │
    │         ▼
    │    build_pipelines()    → StandardScaler + Classifier (inside Pipeline)
    │         │
    │         ▼
    │    cross_validate_all() → 5-fold StratifiedKFold → CV AUC per model
    │         │
    │         ▼
    │    evaluate_on_set()    → Val Accuracy + Val AUC
    │
    ├──► X_val, y_val  ──► select_best() → best model by Val AUC
    │
    └──► X_test, y_test ──► final evaluation (reported once, never used for selection)
                │
                ▼
           best_model.pkl  +  plots/  +  results_summary.json
```

---

## Key Design Decisions

### 1. No Data Leakage
The `StandardScaler` is wrapped **inside** `sklearn.Pipeline`. This means:
- Scaler statistics are computed only on training folds during CV
- Validation and test sets are transformed using training statistics only
- No information from unseen data influences the model

### 2. Proper Evaluation Hierarchy
| Set | Purpose |
|-----|---------|
| Train (60%) | Fit model weights |
| Validation (20%) | Compare models, select best |
| Test (20%) | Single final evaluation — reported in README |

The test set is touched exactly **once**. Using it for model selection would constitute another form of leakage.

### 3. AUC as Primary Metric
Accuracy alone is misleading for medical datasets. AUC (Area Under ROC Curve):
- Is threshold-independent
- Measures ranking quality
- Is more informative when classes are slightly imbalanced

### 4. Config-Driven Hyperparameters
All model parameters live in `configs/config.yaml`. Changing a hyperparameter never requires editing source code — just update the YAML.

---

## Models

| Model | Why included |
|-------|-------------|
| Logistic Regression | Strong linear baseline; interpretable coefficients |
| Random Forest | Handles non-linearity; provides feature importance |
| Gradient Boosting | Often outperforms RF; good with tabular data |
| SVM (RBF) | Effective in medium-dimensional spaces |
| KNN | Simple non-parametric baseline |

---

## Results Summary

After training on the real Kaggle dataset, expected performance:

| Model | CV AUC | Val AUC |
|-------|--------|---------|
| Logistic Regression | ~0.93 | ~0.93 |
| Random Forest | ~0.93 | ~0.93 |
| Gradient Boosting | ~0.94 | ~0.94 |
| SVM | ~0.92 | ~0.92 |
| KNN | ~0.87 | ~0.87 |

> Actual values will appear in `outputs/results_summary.json` after running `train.py`.

---

## Limitations & Future Work

- **Cholesterol = 0** should be imputed (median), not left as-is
- **Hyperparameter tuning** via `GridSearchCV` or `Optuna` would likely improve performance
- **SHAP values** would provide patient-level explainability for clinical use
- **Threshold calibration** — 0.5 default threshold may not be optimal for a medical setting (recall vs precision tradeoff)
- **Cross-dataset validation** — model should be tested on an independent hospital dataset before clinical use
