"""
utils/visualization.py
All plotting functions — clean, consistent style.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

PALETTE = {
    "primary":   "#e63946",
    "secondary": "#2a9d8f",
    "tertiary":  "#e76f51",
    "quaternary":"#457b9d",
    "quinary":   "#8338ec",
    "neutral":   "#6c757d",
}
COLORS = list(PALETTE.values())


def _save(fig, plots_dir: str, filename: str):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✔] Saved: {path}")


def plot_eda(df_raw, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Exploratory Data Analysis", fontsize=14, fontweight="bold", y=1.02)

    # Age distribution
    for target, color, label in [(0, PALETTE["quaternary"], "No Disease"),
                                   (1, PALETTE["primary"],   "Disease")]:
        axes[0].hist(df_raw[df_raw["HeartDisease"] == target]["Age"],
                     bins=20, alpha=0.65, color=color, label=label)
    axes[0].set_title("Age Distribution", fontweight="bold")
    axes[0].set_xlabel("Age"); axes[0].legend()

    # Target balance
    counts = df_raw["HeartDisease"].value_counts().sort_index()
    axes[1].bar(["No Disease", "Disease"], counts.values,
                color=[PALETTE["quaternary"], PALETTE["primary"]], alpha=0.85, width=0.5)
    axes[1].set_title("Target Distribution", fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # MaxHR vs Age
    for target, color, label in [(0, PALETTE["quaternary"], "No Disease"),
                                   (1, PALETTE["primary"],   "Disease")]:
        sub = df_raw[df_raw["HeartDisease"] == target]
        axes[2].scatter(sub["Age"], sub["MaxHR"],
                        alpha=0.45, color=color, label=label, s=18)
    axes[2].set_title("MaxHR vs Age", fontweight="bold")
    axes[2].set_xlabel("Age"); axes[2].set_ylabel("Max Heart Rate")
    axes[2].legend()

    plt.tight_layout()
    _save(fig, plots_dir, "eda.png")


def plot_correlation_heatmap(df_encoded, plots_dir: str):
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.4,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, plots_dir, "correlation_heatmap.png")


def plot_roc_curves(val_results: dict, y_val, plots_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, res), color in zip(val_results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_val, res["y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{name}  (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Validation Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, plots_dir, "roc_curves.png")


def plot_confusion_matrices(pipelines: dict, val_results: dict, y_val, plots_dir: str):
    n = len(pipelines)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, res) in zip(axes, val_results.items()):
        disp = ConfusionMatrixDisplay(res["cm"], display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9, fontweight="bold")
    fig.suptitle("Confusion Matrices — Validation Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, plots_dir, "confusion_matrices.png")


def plot_model_comparison(val_results: dict, cv_results: dict, plots_dir: str):
    names    = list(val_results.keys())
    accs     = [val_results[n]["accuracy"] for n in names]
    aucs     = [val_results[n]["roc_auc"]  for n in names]
    cv_means = [cv_results[n].mean()       for n in names]
    cv_stds  = [cv_results[n].std()        for n in names]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,   accs,     w, label="Val Accuracy",       color=PALETTE["quaternary"], alpha=0.9)
    ax.bar(x,        aucs,    w, label="Val AUC",             color=PALETTE["secondary"],  alpha=0.9)
    ax.bar(x + w, cv_means,   w, yerr=cv_stds, capsize=4,
           label="CV AUC (5-fold)", color=PALETTE["tertiary"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.9, ls="--", color="gray", lw=1, alpha=0.5)
    plt.tight_layout()
    _save(fig, plots_dir, "model_comparison.png")


def plot_feature_importance(clf, feature_names: list, model_name: str, plots_dir: str):
    if not hasattr(clf, "feature_importances_"):
        print(f"[!] {model_name} has no feature_importances_ — skipping.")
        return
    importances = clf.feature_importances_
    idx  = np.argsort(importances)[::-1]
    top  = min(12, len(feature_names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in idx[:top]][::-1],
        importances[idx[:top]][::-1],
        color=PALETTE["primary"], alpha=0.85
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, plots_dir, "feature_importance.png")
