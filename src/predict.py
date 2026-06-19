"""
predict.py
===========
Modul untuk memuat model & preprocessor yang sudah dilatih, lalu melakukan
prediksi harga rumah terhadap input baru (single record maupun batch).

Dipakai langsung oleh app.py (Streamlit) dan juga bisa dipanggil manual:
    python -m src.predict
"""

import json
import joblib
import pandas as pd

from . import config
from . import data_preprocessing as dp


class HousePricePredictor:
    """Wrapper class agar model & preprocessor cukup di-load sekali (efisien untuk Streamlit)."""

    def __init__(self, model_path: str = config.MODEL_PATH,
                 preprocessor_path: str = config.PREPROCESSOR_PATH,
                 metrics_path: str = config.METRICS_PATH):
        self.model = joblib.load(model_path)
        self.preprocessor = dp.load_preprocessor(preprocessor_path)

        try:
            with open(metrics_path, "r") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}

    def get_known_districts(self):
        """Daftar district yang dikenal model (dari data training)."""
        return sorted(self.preprocessor["encoders"]["district_freq_map"].keys())

    def get_known_cities(self):
        """Daftar city yang dikenal model (dari data training)."""
        return self.preprocessor["encoders"]["city_categories"]

    def predict(self, input_dict: dict) -> float:
        """
        Memprediksi harga rumah dari satu input baru.

        Parameters
        ----------
        input_dict : dict, contoh:
            {
                "district": "Kemang",
                "city": "Jakarta Selatan",
                "bed_rooms": 3,
                "bath_rooms": 2,
                "carport": 1,
                "land_area": 150.0,
                "building_area": 180.0,
            }

        Returns
        -------
        float : estimasi harga rumah dalam Rupiah
        """
        df_new = pd.DataFrame([input_dict])
        X_new = dp.preprocess_new_data(df_new, self.preprocessor)
        y_pred_log = self.model.predict(X_new)
        y_pred = dp.inverse_transform_target(y_pred_log)
        return float(y_pred[0])

    def predict_batch(self, df_new: pd.DataFrame) -> pd.Series:
        """Memprediksi harga rumah untuk banyak baris sekaligus (mis. dari file CSV baru)."""
        X_new = dp.preprocess_new_data(df_new, self.preprocessor)
        y_pred_log = self.model.predict(X_new)
        y_pred = dp.inverse_transform_target(y_pred_log)
        return pd.Series(y_pred, index=df_new.index, name="predicted_price")


if __name__ == "__main__":
    predictor = HousePricePredictor()

    sample_input = {
        "district": "Kemang",
        "city": "Jakarta Selatan",
        "bed_rooms": 3,
        "bath_rooms": 2,
        "carport": 1,
        "land_area": 150.0,
        "building_area": 180.0,
    }
    price = predictor.predict(sample_input)
    print(f"Input: {sample_input}")
    print(f"Estimasi harga: Rp {price:,.0f}")
