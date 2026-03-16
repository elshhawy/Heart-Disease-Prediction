"""
src/train.py
Main training entry point — reads config, delegates to utils.
Run: python src/train.py
"""

import os
import sys
import yaml
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.preprocessing import load_data, preprocess, split_data
from src.utils.evaluation    import (
    cross_validate_all, tune_models,
    evaluate_on_set, select_best, save_summary
)
from src.utils.visualization import (
    plot_eda, plot_correlation_heatmap, plot_roc_curves,
    plot_confusion_matrices, plot_model_comparison, plot_feature_importance
)

# ── Load config ────────────────────────────────────────────
CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", cfg["data"]["path"])
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["models_dir"])
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["plots_dir"])
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["outputs_dir"])


def build_pipelines() -> dict:
    mc = cfg["models"]
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(**mc["logistic_regression"]))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(**mc["random_forest"]))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(**mc["gradient_boosting"]))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(**mc["svm"]))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(**mc["knn"]))
        ]),
    }


def main():
    print("=" * 60)
    print("  Heart Disease Prediction — Training Pipeline")
    print("=" * 60 + "\n")

    # 1. Load & preprocess
    df_raw = load_data(DATA_PATH)
    X, y, feature_names = preprocess(df_raw, cfg, MODELS_DIR)

    # 2. Split  →  Train 60% | Val 20% | Test 20%
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)

    # 3. Build base pipelines & cross-validate
    pipelines  = build_pipelines()
    cv_results = cross_validate_all(pipelines, X_train, y_train, cfg)

    # 4. Fit base pipelines on full train set
    for pipe in pipelines.values():
        pipe.fit(X_train, y_train)

    # 5. Evaluate base on Validation set
    print("[ Before Tuning ]")
    val_results_base = evaluate_on_set(pipelines, X_val, y_val, label="Validation")

    # 6. Hyperparameter Tuning (GridSearchCV on train set only)
    tuned_pipelines = tune_models(pipelines, X_train, y_train, cfg)

    # 7. Evaluate tuned on Validation set
    print("[ After Tuning ]")
    val_results = evaluate_on_set(tuned_pipelines, X_val, y_val, label="Validation")

    # 8. Select best model → evaluate ONCE on Test set
    best_name = select_best(tuned_pipelines, val_results, X_test, y_test)

    # 9. Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tuned_pipelines[best_name], os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"\n[✔] Model saved → models/best_model.pkl\n")

    # 10. Plots
    print("── Generating Plots ──")
    plot_eda(df_raw, PLOTS_DIR)
    plot_correlation_heatmap(X.assign(HeartDisease=y), PLOTS_DIR)
    plot_roc_curves(val_results, y_val, PLOTS_DIR)
    plot_confusion_matrices(tuned_pipelines, val_results, y_val, PLOTS_DIR)
    plot_model_comparison(val_results, cv_results, PLOTS_DIR)
    best_clf = tuned_pipelines[best_name].named_steps["clf"]
    plot_feature_importance(best_clf, feature_names, best_name, PLOTS_DIR)

    # 11. Save summary
    save_summary(val_results, cv_results, best_name, OUTPUTS_DIR)

    print("\n[★] Training complete. All artifacts saved.")


if __name__ == "__main__":
    main()
