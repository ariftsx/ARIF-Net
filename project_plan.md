# 🔬 ARIF-Net — Project Plan v1.0

## Pendekatan Baru (Data-Driven Model Development)

> **Filosofi**: Tidak menentukan arsitektur model ARIF-Net di awal. Alur baru:
> **Collection → Preprocessing → Train & Test Baseline → Analisis Gap → Rancang ARIF-Net**

---

## Perubahan dari Project Lama

| Aspek | Project Lama | Project Baru |
|---|---|---|
| **News scraping** | Hanya headline + metadata | **Headline + isi berita** (full body text) |
| **Filter berita** | Filter setelah collection (keyword matching di judul) | **Filter SEBELUM collection** (relevance check) |
| **Scraping method** | BeautifulSoup + custom parser | **Trafilatura** (lebih bersih untuk berita Indonesia) |
| **Media sources** | 6 media | **3 media** (Kompas, Detik, CNN Indonesia) |
| **ARIF-Net model** | Arsitektur ditentukan di awal | **Dikembangkan SETELAH analisis gap** |
| **Structure** | Terpisah per Phase folder | **Satu folder terstruktur pipeline** |

---

## Struktur Folder

```
ARIF-Net/
├── README.md
├── environment.yml
├── requirements.txt
├── run.py                              # Single entry point seluruh pipeline
│
├── configs/
│   ├── project.json                    # Konfigurasi global
│   ├── news_sources.json               # Konfigurasi 3 media
│   ├── news_scope.json                 # Keywords & topics
│   ├── commodities.json                # 3 komoditas PIHPS
│   ├── locations.json                  # Lokasi iklim & pemasok
│   └── models/                         # Config per model baseline
│       ├── persistence.json
│       ├── arima.json
│       ├── ridge.json
│       ├── svr.json
│       ├── rf.json
│       ├── xgb.json
│       ├── lgbm.json
│       ├── lstm.json
│       ├── gru.json
│       ├── bilstm.json
│       ├── dialstm.json
│       ├── tft.json
│       └── tcn_xgb.json
│
├── 01_collection/                      # === TAHAP 1: DATA COLLECTION ===
│   ├── __init__.py
│   ├── collect_pihps.py                # Harga komoditas dari PIHPS BI
│   ├── collect_news.py                 # Berita + isi (trafilatura) dari 3 media
│   ├── collect_climate.py              # Data iklim dari Open-Meteo
│   ├── collect_bbm.py                  # Harga BBM dari Pertamina/MyPertamina
│   ├── collect_calendar.py             # Kalender hari libur & hari besar
│   ├── collect_supply.py               # Data supply (manual/statis)
│   └── utils.py                        # Utilitas bersama (HTTP client, logging)
│
├── 02_preprocessing/                   # === TAHAP 2: PREPROCESSING ===
│   ├── __init__.py
│   ├── preprocess_pihps.py             # Normalisasi harga → target r(t,h)
│   ├── preprocess_news.py              # Sentimen & agregasi berita
│   ├── preprocess_climate.py           # Normalisasi data iklim
│   ├── preprocess_bbm.py              # Normalisasi BBM & logistik
│   ├── preprocess_calendar.py          # Fitur kalender
│   ├── build_target.py                 # Bangun target r(t,h) untuk semua horizon
│   ├── build_features.py              # Gabungkan semua fitur multimodal
│   ├── build_dataset.py               # Bangun dataset final siap model
│   └── utils.py                        # Utilitas preprocessing
│
├── 03_training/                        # === TAHAP 3: TRAINING & EVALUATION ===
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── persistence.py              # Naive baseline (r̂ = 0)
│   │   ├── arima.py                    # ARIMA/SARIMAX
│   │   ├── tabular.py                  # Ridge, SVR, RF, XGBoost, LightGBM
│   │   ├── neural.py                   # LSTM, GRU, BiLSTM, DIA-LSTM, TFT, TCN
│   │   └── twostage.py                 # TCN-XGBoost hybrid
│   ├── data.py                         # DataLoader & batch builder
│   ├── training.py                     # Training loop (shared semua model)
│   ├── metrics.py                      # MAE, RMSE, relMAE, DA, MAPE, R²
│   ├── evaluate.py                     # Evaluasi & perbandingan model
│   └── analyze_gaps.py                 # Analisis kelemahan → novelty ARIF-Net
│
├── 04_arif_net/                        # === TAHAP 4: ARIF-Net (setelah analisis) ===
│   ├── __init__.py
│   └── README.md                       # Placeholder — diisi setelah gap analysis
│
├── data/
│   ├── raw/                            # Data mentah dari collection
│   │   ├── pihps/
│   │   ├── news/
│   │   ├── climate/
│   │   ├── bbm/
│   │   ├── calendar/
│   │   └── supply/
│   ├── processed/                      # Data hasil preprocessing
│   └── final/                          # Dataset final siap model
│
└── results/
    ├── runs/                           # Output training per run
    ├── evaluation/                     # Tabel & grafik perbandingan
    └── gap_analysis/                   # Hasil analisis gap & rekomendasi
```

## Pipeline Execution Order

```
python run.py collect     # Tahap 1: Collection semua data
python run.py preprocess  # Tahap 2: Preprocessing & build dataset
python run.py train       # Tahap 3: Training semua baseline
python run.py evaluate    # Tahap 3: Evaluasi & perbandingan
python run.py analyze     # Tahap 3: Gap analysis → rekomendasi ARIF-Net
```

## Model Baseline (dari Penelitian Terdahulu)

| # | Model | Referensi | Tipe |
|---|---|---|---|
| 1 | Persistence | Naive baseline | Statistik |
| 2 | ARIMA | Klasik time series | Statistik |
| 3 | Ridge Regression | Baseline linear | ML Tabular |
| 4 | SVR | Support Vector Regression | ML Tabular |
| 5 | Random Forest | Ensemble tree | ML Tabular |
| 6 | XGBoost | Gradient boosting | ML Tabular |
| 7 | LightGBM | Gradient boosting | ML Tabular |
| 8 | LSTM | Manogna et al. (2025) | Deep Learning |
| 9 | GRU | Manogna et al. (2025) | Deep Learning |
| 10 | BiLSTM | Baseline deep multimodal | Deep Learning |
| 11 | DIA-LSTM | Gu et al. (2022) | Deep Learning |
| 12 | TFT | Amalia et al. (2026) | Deep Learning |
| 13 | TCN-XGBoost | Zhao et al. (2025) | Hybrid |

## Komoditas & Horizon

- **Komoditas**: Cabai Merah Keriting (com_14), Bawang Merah (com_11), Beras Medium I (com_3)
- **Pasar**: Pasar Kramatjati, Jakarta — harga eceran PIHPS BI
- **Horizon**: 1, 3, 7, 14, 30, 90, 180, 365 hari
- **Periode**: 2019-01-01 → 2026-09-30
- **Validasi**: Walk-forward 3-fold kronologis; test buta terpisah

## Metrik Evaluasi

- MAE, RMSE pada target r(t,h)
- relMAE (relatif terhadap persistence)
- Directional Accuracy (DA)
- MAPE pada harga rekonstruksi
- R²
- Diebold-Mariano test (signifikansi statistik)
