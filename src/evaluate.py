"""
evaluate.py
============
Modul untuk menghitung metrik evaluasi model regresi (MAE, RMSE, R2)
serta membuat visualisasi sederhana (actual vs predicted, feature importance).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # agar bisa generate plot tanpa display/GUI
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Menghitung MAE, RMSE, dan R2 pada skala harga asli (bukan skala log)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_metrics_table(results: dict) -> None:
    """Menampilkan tabel perbandingan metrik antar model ke console."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20}{'MAE':>15}{'RMSE':>18}{'R2 Score':>12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20}{metrics['mae']:>15,.0f}{metrics['rmse']:>18,.0f}{metrics['r2']:>12.4f}")
    print("=" * 70)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, save_dir: str = config.PLOT_DIR):
    """Scatter plot actual vs predicted price untuk satu model."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2563eb")
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Prediksi Sempurna")
    ax.set_xlabel("Harga Aktual (Rp)")
    ax.set_ylabel("Harga Prediksi (Rp)")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, f"actual_vs_predicted_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, model_name: str, save_dir: str = config.PLOT_DIR):
    """Plot feature importance untuk model tree-based (RandomForest, GradientBoosting, XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        return None

    os.makedirs(save_dir, exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#16a34a",
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance - {model_name}")
    fig.tight_layout()
    path = os.path.join(save_dir, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(results: dict, save_dir: str = config.PLOT_DIR):
    """Bar chart perbandingan R2 score seluruh model."""
    os.makedirs(save_dir, exist_ok=True)
    names = list(results.keys())
    r2_scores = [results[n]["r2"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, r2_scores, color="#7c3aed")
    ax.set_ylabel("R2 Score")
    ax.set_title("Perbandingan R2 Score Antar Model")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center")
    fig.tight_layout()
    path = os.path.join(save_dir, "model_comparison_r2.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
