# 🏠 House Price Forecasting — Jakarta

Project Data Science end-to-end untuk **prediksi harga rumah di Jakarta**, mulai dari preprocessing data, training & perbandingan model, evaluasi, hingga **deployment** menggunakan **Streamlit**.

Project ini disusun dengan struktur ala *research project / library* (bukan satu notebook tunggal) agar mudah dikembangkan, diuji, dan di-deploy.

---

## 📁 Struktur Project

```
house_price_forecasting/
│
├── data/
│   └── dataset.csv              # dataset mentah (10.000 baris, properti Jakarta)
│
├── notebooks/
│   └── exploration.ipynb        # EDA awal (distribusi, korelasi, missing value)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # semua path & parameter terpusat di sini
│   ├── data_preprocessing.py     # load, clean, encode, scale data
│   ├── train.py                  # training & perbandingan model
│   ├── evaluate.py               # metrik & visualisasi evaluasi
│   └── predict.py                # load model & prediksi data baru
│
├── models/
│   ├── model.pkl                 # model terbaik (hasil train.py)
│   ├── preprocessor.pkl          # encoder + scaler yang sudah di-fit
│   ├── metrics.json              # ringkasan perbandingan semua model
│   └── plots/                    # visualisasi hasil evaluasi (.png)
│
├── app.py                        # aplikasi Streamlit (deployment)
├── requirements.txt
└── README.md
```

---

## 📊 Tentang Dataset

Dataset berisi **10.000 baris** data properti di 5 wilayah Jakarta dengan kolom:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `price` | numerik | **Target** — harga rumah (Rupiah) |
| `district` | kategorikal | Kawasan/kelurahan (253 kategori unik) |
| `city` | kategorikal | Wilayah kota (5 kategori: Jakarta Selatan/Barat/Utara/Timur/Pusat) |
| `bed_rooms` | numerik | Jumlah kamar tidur |
| `bath_rooms` | numerik | Jumlah kamar mandi |
| `carport` | numerik | Jumlah carport |
| `land_area` | numerik | Luas tanah (m²), terdapat sedikit missing value |
| `building_area` | numerik | Luas bangunan (m²), terdapat sedikit missing value |

### Insight awal (lihat `notebooks/exploration.ipynb` untuk detail)
- `price` **sangat skewed** (rentang ~Rp1,8 juta s/d ~Rp900 miliar) → ditangani dengan **log1p transform** saat training.
- `district` memiliki kardinalitas tinggi (253 kategori) → menggunakan **frequency encoding**, bukan one-hot (agar dimensi fitur tidak meledak).
- `city` hanya 5 kategori → aman menggunakan **one-hot encoding**.
- Terdapat outlier ekstrem pada `price` → di-*cap* (winsorizing) pada persentil 0.5%–99.5% agar baseline model linear lebih stabil.

---

## ⚙️ Pipeline Preprocessing (`src/data_preprocessing.py`)

1. **Cleaning**: drop kolom `index`, drop duplikat, imputasi missing value (`land_area`, `building_area`) dengan median, clip nilai negatif.
2. **Outlier capping**: winsorizing pada `price` (dapat dimatikan/diubah via `config.py`).
3. **Encoding**:
   - `city` → One-Hot Encoding
   - `district` → Frequency Encoding (mencegah dimensi tinggi & overfitting)
4. **Scaling**: `StandardScaler` pada seluruh fitur numerik.
5. **Target transform**: `log1p(price)` saat training, `expm1()` saat inverse (prediksi dikembalikan ke skala Rupiah asli).

Seluruh object preprocessing (encoder + scaler) disimpan ke `models/preprocessor.pkl` agar transformasi pada data baru **konsisten** dengan saat training (tidak ada data leakage).

---

## 🤖 Model yang Dibandingkan (`src/train.py`)

| Model | Keterangan |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Ensemble tree-based |
| Gradient Boosting Regressor | Boosting (scikit-learn) |
| XGBoost Regressor | Boosting (jika library `xgboost` terinstall) |

Model dievaluasi dengan **MAE**, **RMSE**, dan **R² Score** (dihitung pada skala harga asli, bukan skala log). Model dengan **R² tertinggi** otomatis dipilih dan disimpan sebagai `models/model.pkl`.

### Hasil training (run terakhir di environment ini)

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 5.633.277.538 | 19.098.143.760 | -0.321 |
| Random Forest | 2.943.550.402 | 7.561.023.203 | 0.793 |
| **Gradient Boosting** ✅ | **2.947.923.087** | **7.469.983.394** | **0.798** |
| XGBoost | 2.864.361.844 | 7.539.714.444 | 0.794 |

> Baseline Linear Regression berkinerja buruk (R² negatif) karena hubungan harga rumah dengan fiturnya sangat non-linear & dipengaruhi outlier ekstrem — ini wajar dan menjadi alasan model tree-based/boosting dipilih sebagai model produksi.

---

## 🚀 Cara Menjalankan Project

### 1. Persiapan environment

```bash
cd house_price_forecasting
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pastikan dataset tersedia

Letakkan dataset Anda di `data/dataset.csv` (kolom harus sesuai struktur di atas).

### 3. Jalankan training

```bash
python -m src.train
```

Perintah ini akan:
- Memuat & menampilkan struktur dataset
- Melakukan preprocessing
- Melatih & membandingkan 4 model
- Menyimpan model terbaik ke `models/model.pkl`
- Menyimpan preprocessor ke `models/preprocessor.pkl`
- Menyimpan ringkasan metrik ke `models/metrics.json`
- Menyimpan visualisasi evaluasi ke `models/plots/`

### 4. (Opsional) Coba prediksi lewat terminal

```bash
python -m src.predict
```

### 5. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`, lalu isi form:
- Kota & lokasi/kawasan
- Jumlah kamar tidur & kamar mandi
- Jumlah carport
- Luas tanah & luas bangunan

Aplikasi akan menampilkan:
- 💰 Estimasi harga rumah
- ℹ️ Informasi model (model terpakai, R², MAE, RMSE, perbandingan antar model)
- 📊 Visualisasi posisi estimasi harga terhadap distribusi harga pasar

### 6. (Opsional) Eksplorasi data lebih lanjut

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🧩 Catatan Pengembangan Lanjutan

- Untuk dataset baru dengan kolom berbeda, cukup sesuaikan `CATEGORICAL_LOW_CARD`, `CATEGORICAL_HIGH_CARD`, dan `NUMERIC_COLS` di `src/config.py` — tidak perlu mengubah logika di `data_preprocessing.py`.
- Hyperparameter setiap model dapat diatur di `MODEL_PARAMS` (`src/config.py`).
- Metrik pemilihan model terbaik (`SELECTION_METRIC`) dapat diganti ke `"mae"` atau `"rmse"` sesuai kebutuhan bisnis.
- Untuk retrain dengan data baru, cukup ganti `data/dataset.csv` lalu jalankan ulang `python -m src.train` — `model.pkl` & `preprocessor.pkl` akan ter-overwrite otomatis.

---

## 👤 Author

Project ini dibuat sebagai tugas mata kuliah **Data Science — Deployment**, Program Studi Sistem Informasi.
