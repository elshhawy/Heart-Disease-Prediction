"""
utils/evaluation.py
Model evaluation: cross-validation, hyperparameter tuning, full metrics, summary.
"""

import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)


# ── Hyperparameter grids ───────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C":        [0.01, 0.1, 1, 10],
        "clf__solver":   ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "clf__n_estimators":     [100, 200, 300],
        "clf__max_depth":        [5, 10, 15, None],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "clf__n_estimators":  [100, 200],
        "clf__max_depth":     [3, 4, 5],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    },
    "SVM": {
        "clf__C":      [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights":     ["uniform", "distance"],
    },
}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return all classification metrics as a dict."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba),                   4),
    }


def cross_validate_all(pipelines: dict, X_train, y_train, cfg: dict) -> dict:
    """5-fold CV AUC for all pipelines."""
    cv = StratifiedKFold(
        n_splits=cfg["training"]["cv_folds"],
        shuffle=True,
        random_state=cfg["training"]["random_state"]
    )
    cv_results = {}
    print(f"── Cross-Validation ({cfg['training']['cv_folds']}-fold) ──")
    for name, pipe in pipelines.items():
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_results[name] = scores
        print(f"  [{name}]  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print()
    return cv_results


def tune_models(pipelines: dict, X_train, y_train, cfg: dict) -> dict:
    """GridSearchCV on each pipeline. Returns dict of tuned pipelines."""
    cv = StratifiedKFold(
        n_splits=cfg["training"]["cv_folds"],
        shuffle=True,
        random_state=cfg["training"]["random_state"]
    )
    tuned = {}
    print("── Hyperparameter Tuning ──")
    for name, pipe in pipelines.items():
        grid = PARAM_GRIDS.get(name, {})
        if not grid:
            tuned[name] = pipe
            print(f"  [{name}]  no grid — using defaults")
            continue
        gs = GridSearchCV(
            pipe, grid, cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True
        )
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        print(f"  [{name}]  best AUC: {gs.best_score_:.4f}  |  {gs.best_params_}")
    print()
    return tuned


def evaluate_on_set(pipelines: dict, X, y, label="Validation") -> dict:
    """Evaluate all pipelines — returns full metrics dict."""
    results = {}
    print(f"── {label} Set ──")
    for name, pipe in pipelines.items():
        y_pred  = pipe.predict(X)
        y_proba = pipe.predict_proba(X)[:, 1]
        m = compute_metrics(y, y_pred, y_proba)
        results[name] = {
            **m,
            "cm":      confusion_matrix(y, y_pred),
            "report":  classification_report(y, y_pred, output_dict=True),
            "y_pred":  y_pred,
            "y_proba": y_proba,
        }
        print(f"  [{name}]  acc: {m['accuracy']:.4f} | "
              f"prec: {m['precision']:.4f} | "
              f"rec: {m['recall']:.4f} | "
              f"f1: {m['f1']:.4f} | "
              f"auc: {m['roc_auc']:.4f}")
    print()
    return results


def select_best(pipelines: dict, val_results: dict, X_test, y_test) -> str:
    """Pick best by val AUC → evaluate once on test set."""
    best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])
    best_pipe  = pipelines[best_name]

    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, y_pred, y_proba)

    print("── Final Test Set ──")
    print(f"  ★  Best model : {best_name}")
    print(f"     Accuracy   : {m['accuracy']:.4f}")
    print(f"     Precision  : {m['precision']:.4f}")
    print(f"     Recall     : {m['recall']:.4f}")
    print(f"     F1         : {m['f1']:.4f}")
    print(f"     AUC        : {m['roc_auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    return best_name


def save_summary(val_results: dict, cv_results: dict,
                 best_name: str, outputs_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)
    summary = {}
    for name in val_results:
        summary[name] = {
            "val_accuracy":  val_results[name]["accuracy"],
            "val_precision": val_results[name]["precision"],
            "val_recall":    val_results[name]["recall"],
            "val_f1":        val_results[name]["f1"],
            "val_auc":       val_results[name]["roc_auc"],
            "cv_auc_mean":   round(cv_results[name].mean(), 4),
            "cv_auc_std":    round(cv_results[name].std(),  4),
        }
    summary["best_model"] = best_name
    path = os.path.join(outputs_dir, "results_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✔] Saved: {path}")
