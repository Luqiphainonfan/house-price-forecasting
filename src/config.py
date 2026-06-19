"""
config.py
==========
Konfigurasi terpusat untuk seluruh pipeline (preprocessing, training, evaluasi, deployment).
Mengubah parameter project cukup dilakukan di file ini.
"""

import os

# ---------------------------------------------------------------------------
# PATH PROJECT
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")              # model terbaik (final)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")  # encoder + scaler + metadata
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")          # ringkasan hasil evaluasi semua model
PLOT_DIR = os.path.join(MODEL_DIR, "plots")

# ---------------------------------------------------------------------------
# DEFINISI KOLOM DATASET
# ---------------------------------------------------------------------------
TARGET_COL = "price"

ID_COLS = ["index"]                      # kolom yang dibuang (tidak informatif)
CATEGORICAL_LOW_CARD = ["city"]          # one-hot encoding
CATEGORICAL_HIGH_CARD = ["district"]     # frequency encoding
NUMERIC_COLS = ["bed_rooms", "bath_rooms", "carport", "land_area", "building_area"]

# Kolom-kolom numerik yang boleh memiliki missing value & akan diimputasi median
NUMERIC_COLS_WITH_NA = ["land_area", "building_area"]

# ---------------------------------------------------------------------------
# PARAMETER PREPROCESSING
# ---------------------------------------------------------------------------
# Target (price) dilog-transform karena distribusinya sangat skewed (min ~1.8jt, max ~900M)
USE_LOG_TARGET = True

# Capping outlier ekstrem pada price berdasarkan persentil (mengurangi pengaruh outlier
# terhadap model linear, tanpa membuang data — hanya membatasi nilai ekstremnya / winsorizing)
CAP_OUTLIERS = True
PRICE_LOWER_QUANTILE = 0.005
PRICE_UPPER_QUANTILE = 0.995

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# PARAMETER MODEL
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}

# Metrik utama yang dipakai untuk memilih model terbaik
SELECTION_METRIC = "r2"  # salah satu dari: "mae", "rmse", "r2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
