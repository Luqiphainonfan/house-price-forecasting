"""
app.py
=======
Aplikasi Streamlit untuk estimasi harga rumah Jakarta.
Menggunakan model & preprocessor yang sudah dilatih lewat src/train.py.

Cara menjalankan:
    streamlit run app.py
"""

import json
import pandas as pd
import streamlit as st

from src import config
from src.predict import HousePricePredictor

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Estimasi Harga Rumah Jakarta",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_predictor():
    """Load model & preprocessor sekali saja (di-cache oleh Streamlit)."""
    return HousePricePredictor()


def main():
    st.title("🏠 Estimasi Harga Rumah Jakarta")
    st.caption(
        "Prediksi harga rumah berdasarkan lokasi, luas tanah/bangunan, dan jumlah ruangan, "
        "menggunakan model Machine Learning yang dilatih dari data properti Jakarta."
    )

    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error(
            "Model belum ditemukan. Jalankan training terlebih dahulu dengan perintah:\n\n"
            "`python -m src.train`"
        )
        st.stop()

    districts = predictor.get_known_districts()
    cities = predictor.get_known_cities()

    # -----------------------------------------------------------------
    # Sidebar: Informasi model
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("ℹ️ Informasi Model")
        meta = predictor.metadata
        if meta:
            best_model = meta.get("best_model", "-")
            metrics = meta.get("all_results", {}).get(best_model, {})
            st.markdown(f"**Model digunakan:** `{best_model}`")
            if metrics:
                st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                st.metric("MAE", f"Rp {metrics.get('mae', 0):,.0f}")
                st.metric("RMSE", f"Rp {metrics.get('rmse', 0):,.0f}")

            with st.expander("Bandingkan semua model"):
                df_compare = pd.DataFrame(meta.get("all_results", {})).T
                df_compare.columns = ["MAE", "RMSE", "R2 Score"]
                st.dataframe(df_compare.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2 Score": "{:.3f}"}))
        else:
            st.info("Metadata model tidak ditemukan.")

        st.divider()
        st.caption("Dataset: properti Jakarta (5 wilayah, 253 kawasan/district).")

    # -----------------------------------------------------------------
    # Form input user
    # -----------------------------------------------------------------
    st.subheader("Masukkan Detail Rumah")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Kota", options=cities)
        district = st.selectbox("Lokasi / Kawasan (district)", options=districts)
        bed_rooms = st.number_input("Jumlah Kamar Tidur", min_value=0, max_value=20, value=3, step=1)
        bath_rooms = st.number_input("Jumlah Kamar Mandi", min_value=0, max_value=20, value=2, step=1)

    with col2:
        carport = st.number_input("Jumlah Carport", min_value=0, max_value=10, value=1, step=1)
        land_area = st.number_input("Luas Tanah (m²)", min_value=1.0, max_value=5000.0, value=120.0, step=1.0)
        building_area = st.number_input("Luas Bangunan (m²)", min_value=1.0, max_value=5000.0, value=150.0, step=1.0)

    predict_btn = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

    # -----------------------------------------------------------------
    # Hasil prediksi
    # -----------------------------------------------------------------
    if predict_btn:
        input_dict = {
            "district": district,
            "city": city,
            "bed_rooms": bed_rooms,
            "bath_rooms": bath_rooms,
            "carport": carport,
            "land_area": land_area,
            "building_area": building_area,
        }

        with st.spinner("Menghitung estimasi harga..."):
            price = predictor.predict(input_dict)

        st.success("Estimasi berhasil dihitung!")
        st.metric("💰 Estimasi Harga Rumah", f"Rp {price:,.0f}")

        # Visualisasi sederhana: posisi harga estimasi relatif terhadap range harga di dataset
        st.subheader("📊 Visualisasi")
        try:
            df_raw = pd.read_csv(config.RAW_DATA_PATH)
            fig_data = df_raw["price"].clip(upper=df_raw["price"].quantile(0.99))

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(fig_data, bins=50, color="#93c5fd", edgecolor="white")
            ax.axvline(price, color="red", linestyle="--", linewidth=2, label="Estimasi Anda")
            ax.set_xlabel("Harga (Rp)")
            ax.set_ylabel("Jumlah Properti")
            ax.set_title("Posisi Estimasi Harga terhadap Distribusi Harga Pasar")
            ax.legend()
            st.pyplot(fig)
        except Exception:
            st.info("Visualisasi tidak tersedia.")

        with st.expander("Lihat detail input"):
            st.json(input_dict)


if __name__ == "__main__":
    main()
