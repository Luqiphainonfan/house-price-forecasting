"""
train.py
=========
Script utama untuk:
1. Memuat & melakukan preprocessing data
2. Melatih beberapa model regresi (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
3. Mengevaluasi & membandingkan performa tiap model (MAE, RMSE, R2)
4. Menyimpan model terbaik + preprocessor ke folder models/

Cara menjalankan (dari root folder project):
    python -m src.train
"""

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from . import config
from . import data_preprocessing as dp
from . import evaluate as ev


def get_models() -> dict:
    """Mendefinisikan seluruh model yang akan dibandingkan."""
    models = {
        "linear_regression": LinearRegression(**config.MODEL_PARAMS["linear_regression"]),
        "random_forest": RandomForestRegressor(**config.MODEL_PARAMS["random_forest"]),
        "gradient_boosting": GradientBoostingRegressor(**config.MODEL_PARAMS["gradient_boosting"]),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(**config.MODEL_PARAMS["xgboost"])
    else:
        print("[WARNING] xgboost tidak terinstall, model xgboost dilewati. "
              "Install dengan: pip install xgboost")
    return models


def train_and_compare():
    print("\n[1/5] Memuat & inspeksi dataset ...")
    df = dp.load_data()
    dp.inspect_data(df)

    print("\n[2/5] Preprocessing data (cleaning, encoding, scaling) ...")
    X, y, preprocessor, feature_names = dp.preprocess_pipeline(df)
    print(f"Jumlah fitur final: {len(feature_names)} -> {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    print("\n[3/5] Training & evaluasi tiap model ...")
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"  -> Training {name} ...")
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        # Inverse transform ke skala harga asli (Rupiah) sebelum dihitung metriknya
        y_pred = dp.inverse_transform_target(y_pred_log)
        y_true = dp.inverse_transform_target(y_test)

        metrics = ev.compute_metrics(y_true, y_pred)
        results[name] = metrics
        trained_models[name] = model

        ev.plot_actual_vs_predicted(y_true, y_pred, name)
        ev.plot_feature_importance(model, feature_names, name)

    ev.print_metrics_table(results)
    ev.plot_model_comparison(results)

    print("\n[4/5] Memilih model terbaik ...")
    metric = config.SELECTION_METRIC
    if metric == "r2":
        best_name = max(results, key=lambda k: results[k]["r2"])
    else:  # mae atau rmse -> semakin kecil semakin baik
        best_name = min(results, key=lambda k: results[k][metric])

    best_model = trained_models[best_name]
    print(f"Model terbaik berdasarkan '{metric}': {best_name} -> {results[best_name]}")

    print("\n[5/5] Menyimpan model & preprocessor ...")
    joblib.dump(best_model, config.MODEL_PATH)
    dp.save_preprocessor(preprocessor, config.PREPROCESSOR_PATH)

    # Simpan ringkasan metrik + metadata model terbaik agar bisa dipakai di app.py
    summary = {
        "best_model": best_name,
        "all_results": results,
        "feature_names": feature_names,
        "selection_metric": metric,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Model disimpan di: {config.MODEL_PATH}")
    print(f"Preprocessor disimpan di: {config.PREPROCESSOR_PATH}")
    print(f"Metrics summary disimpan di: {config.METRICS_PATH}")
    print("\nSelesai. Project siap untuk tahap deployment (app.py).")

    return best_model, preprocessor, summary


if __name__ == "__main__":
    train_and_compare()
