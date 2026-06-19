"""
data_preprocessing.py
======================
Modul untuk:
1. Memuat dataset mentah
2. Menampilkan struktur dataset (info, missing value, statistik)
3. Membersihkan data (handle missing value, outlier capping)
4. Melakukan encoding kategori (one-hot untuk low-cardinality, frequency untuk high-cardinality)
5. Melakukan scaling fitur numerik
6. Menyimpan & memuat object preprocessing (agar konsisten dipakai ulang saat predict/deploy)

Semua fungsi didesain modular sehingga bisa dipakai ulang baik di train.py, evaluate.py,
predict.py, maupun app.py (Streamlit) tanpa duplikasi logika.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


# ---------------------------------------------------------------------------
# 1. LOAD & INSPEKSI DATA
# ---------------------------------------------------------------------------
def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Menampilkan ringkasan struktur dataset: shape, dtype, missing value, statistik dasar."""
    print("=" * 60)
    print(f"Shape dataset : {df.shape}")
    print("=" * 60)
    print("\nTipe data per kolom:")
    print(df.dtypes)
    print("\nJumlah missing value per kolom:")
    print(df.isnull().sum())
    print("\nStatistik deskriptif:")
    print(df.describe(include="all").T)


# ---------------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data mentah:
    - Drop kolom ID yang tidak informatif
    - Drop duplikat baris
    - Imputasi missing value numerik dengan median
    - Capping outlier ekstrem pada target price (winsorizing), opsional via config
    """
    df = df.copy()

    # Drop kolom ID
    drop_cols = [c for c in config.ID_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop duplikat baris
    df = df.drop_duplicates()

    # Imputasi missing value numerik dengan median
    for col in config.NUMERIC_COLS_WITH_NA:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Pastikan tidak ada nilai negatif pada fitur numerik (data fisik tidak mungkin negatif)
    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Capping outlier ekstrem pada price agar model (khususnya linear) lebih stabil
    if config.CAP_OUTLIERS and config.TARGET_COL in df.columns:
        lower = df[config.TARGET_COL].quantile(config.PRICE_LOWER_QUANTILE)
        upper = df[config.TARGET_COL].quantile(config.PRICE_UPPER_QUANTILE)
        df[config.TARGET_COL] = df[config.TARGET_COL].clip(lower=lower, upper=upper)

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. ENCODING
# ---------------------------------------------------------------------------
def fit_district_frequency_map(df: pd.DataFrame, col: str = "district") -> dict:
    """Menghitung frequency encoding map (proporsi kemunculan) dari kolom high-cardinality."""
    freq = df[col].value_counts(normalize=True)
    return freq.to_dict()


def apply_district_frequency(df: pd.DataFrame, freq_map: dict, col: str = "district") -> pd.Series:
    """
    Menerapkan frequency encoding. Kategori yang belum pernah terlihat saat fit
    (misal input baru dari user di Streamlit) diberi nilai minimum dari freq_map (fallback aman).
    """
    fallback = min(freq_map.values()) if len(freq_map) > 0 else 0.0
    return df[col].map(freq_map).fillna(fallback)


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Melakukan encoding terhadap kolom kategorikal:
    - city       -> One-Hot Encoding
    - district   -> Frequency Encoding

    Parameters
    ----------
    df : DataFrame hasil clean_data()
    encoders : dict berisi {'district_freq_map':..., 'city_categories':...}.
               Wajib diisi saat fit=False (mode transform/predict).
    fit : True saat training (membangun encoder baru), False saat predict (re-use encoder lama)

    Returns
    -------
    df_encoded : DataFrame numerik siap dipakai model
    encoders   : dict encoder (disimpan agar bisa dipakai ulang saat predict)
    """
    df = df.copy()
    encoders = {} if encoders is None else encoders

    # --- Frequency encoding untuk district ---
    if fit:
        district_freq_map = fit_district_frequency_map(df, "district")
        encoders["district_freq_map"] = district_freq_map
    else:
        district_freq_map = encoders["district_freq_map"]

    df["district_freq"] = apply_district_frequency(df, district_freq_map, "district")
    df = df.drop(columns=["district"])

    # --- One-hot encoding untuk city ---
    if fit:
        city_categories = sorted(df["city"].unique().tolist())
        encoders["city_categories"] = city_categories
    else:
        city_categories = encoders["city_categories"]

    for cat in city_categories:
        df[f"city_{cat}"] = (df["city"] == cat).astype(int)
    df = df.drop(columns=["city"])

    # Susun urutan kolom final secara konsisten (penting agar urutan fitur sama persis
    # antara saat training dan saat predict)
    feature_cols = config.NUMERIC_COLS + ["district_freq"] + [f"city_{c}" for c in city_categories]
    encoders["feature_columns"] = feature_cols

    df_encoded = df[feature_cols].copy()
    return df_encoded, encoders


# ---------------------------------------------------------------------------
# 4. SCALING
# ---------------------------------------------------------------------------
def scale_features(X: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """Melakukan standardisasi (zero mean, unit variance) pada seluruh fitur numerik."""
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# 5. TARGET TRANSFORM
# ---------------------------------------------------------------------------
def transform_target(y: pd.Series) -> np.ndarray:
    """log1p transform target price (mengurangi efek skewness ekstrem)."""
    return np.log1p(y.values) if config.USE_LOG_TARGET else y.values


def inverse_transform_target(y_pred: np.ndarray) -> np.ndarray:
    """Mengembalikan prediksi ke skala harga asli (Rupiah)."""
    return np.expm1(y_pred) if config.USE_LOG_TARGET else y_pred


# ---------------------------------------------------------------------------
# 6. PIPELINE UTAMA (orkestrasi semua langkah di atas)
# ---------------------------------------------------------------------------
def preprocess_pipeline(df: pd.DataFrame):
    """
    Pipeline lengkap dari data mentah -> X (scaled), y (log-transformed), preprocessor objects.
    Dipakai di train.py.
    """
    df_clean = clean_data(df)

    y_raw = df_clean[config.TARGET_COL]
    X_raw = df_clean.drop(columns=[config.TARGET_COL])

    X_encoded, encoders = encode_features(X_raw, fit=True)
    X_scaled, scaler = scale_features(X_encoded, fit=True)
    y = transform_target(y_raw)

    preprocessor = {
        "encoders": encoders,
        "scaler": scaler,
    }
    return X_scaled, y, preprocessor, X_encoded.columns.tolist()


def preprocess_new_data(df_new: pd.DataFrame, preprocessor: dict):
    """
    Preprocessing untuk data baru (saat predict/deploy), menggunakan encoder & scaler
    yang sudah di-fit sebelumnya saat training (tidak fit ulang -> mencegah data leakage).
    """
    df = df_new.copy()

    # Imputasi missing value memakai aturan yang sama (median dihitung dari train,
    # namun untuk single-row input dari user biasanya tidak ada NA -> aman)
    for col in config.NUMERIC_COLS_WITH_NA:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].isnull().all() is False else 0)

    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    X_encoded, _ = encode_features(df, encoders=preprocessor["encoders"], fit=False)
    X_scaled, _ = scale_features(X_encoded, scaler=preprocessor["scaler"], fit=False)
    return X_scaled


# ---------------------------------------------------------------------------
# 7. SAVE / LOAD PREPROCESSOR
# ---------------------------------------------------------------------------
def save_preprocessor(preprocessor: dict, path: str = config.PREPROCESSOR_PATH) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str = config.PREPROCESSOR_PATH) -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    # Quick check saat file dijalankan langsung: python -m src.data_preprocessing
    df = load_data()
    inspect_data(df)
