# ARIF-Net — Agricultural Risk and Inflation Forecasting Network

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Multimodal Regression Forecasting untuk Prediksi Perubahan Harga Komoditas Pangan Indonesia**

Proyek penelitian untuk memprediksi perubahan harga komoditas pangan strategis Indonesia
menggunakan pendekatan multimodal yang menggabungkan data harga historis, iklim, berita,
harga BBM, dan kalender. Dikembangkan sebagai bagian dari penelitian di Program Studi
Teknik Informatika, Universitas Mercu Buana.

---

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Pipeline](#arsitektur-pipeline)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Struktur Direktori](#struktur-direktori)
- [Sumber Data](#sumber-data)
- [Model Baseline](#model-baseline)
- [Metrik Evaluasi](#metrik-evaluasi)
- [Dokumentasi Lanjutan](#dokumentasi-lanjutan)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

---

## Pendahuluan

Fluktuasi harga komoditas pangan di Indonesia, khususnya cabai merah, bawang merah,
dan beras, berdampak signifikan terhadap inflasi dan kesejahteraan masyarakat.
Proyek ini mengembangkan sistem forecasting yang:

1. **Mengumpulkan data multimodal** dari berbagai sumber resmi dan publik
2. **Memproses dan mengintegrasikan** fitur dari 6 modalitas berbeda
3. **Melatih dan mengevaluasi** 13 model dari literatur terdahulu
4. **Menganalisis kelemahan** (gap analysis) untuk merancang model baru (ARIF-Net)

### Pendekatan: Data-Driven Model Development

```
Collection --> Preprocessing --> Train Baseline --> Gap Analysis --> Design ARIF-Net
```

Arsitektur model ARIF-Net **tidak ditentukan di awal**. Alur pengembangan dimulai dari
pengumpulan data, pelatihan model-model baseline dari penelitian terdahulu, analisis
kelemahan, dan baru kemudian merancang arsitektur model baru berdasarkan temuan empiris.

---

## Fitur Utama

- **6 modalitas data**: harga historis, iklim, berita (headline + isi), BBM, kalender, supply
- **3 komoditas target**: Cabai Merah Keriting, Bawang Merah, Beras Medium I
- **8 horizon prediksi**: 1, 3, 7, 14, 30, 90, 180, 365 hari
- **13 model baseline**: dari statistik klasik hingga deep learning
- **Scraping berita lengkap**: mengambil isi artikel (bukan hanya headline) dengan filter relevansi
- **Pipeline terotomasi**: satu perintah (`run.py`) untuk menjalankan seluruh tahap
- **Gap analysis otomatis**: mengidentifikasi kelemahan model dan merekomendasikan arsitektur baru

---

## Arsitektur Pipeline

```
+-------------------+     +---------------------+     +---------------------+
|  01_collection    |     |  02_preprocessing   |     |  03_training        |
|                   |     |                     |     |                     |
|  - PIHPS (harga)  | --> |  - Normalisasi      | --> |  - 13 model         |
|  - News (3 media) |     |  - Sentimen berita   |     |  - Evaluasi         |
|  - Climate (18kab)|     |  - Target r(t,h)    |     |  - Gap analysis     |
|  - BBM            |     |  - Feature merging   |     |                     |
|  - Calendar       |     |  - Train/Val/Test   |     |                     |
+-------------------+     +---------------------+     +----------+----------+
                                                                 |
                                                                 v
                                                      +---------------------+
                                                      |  04_arif_net        |
                                                      |  (setelah gap       |
                                                      |   analysis selesai) |
                                                      +---------------------+
```

---

## Persyaratan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| OS | Windows 10 / Linux | Windows 11 / Ubuntu 22.04 |
| Python | 3.12 | 3.12+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB (termasuk data) | 10 GB |
| GPU | Tidak diperlukan (CPU-only) | - |
| Internet | Diperlukan (untuk collection) | Koneksi stabil |

---

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/ariftsx/ARIF-Net.git
cd ARIF-Net
```

### 2. Buat Environment (Conda)

```bash
conda env create -f environment.yml
conda activate ARIF-Net
```

### 3. Atau Gunakan pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Verifikasi Instalasi

```bash
python run.py status
```

Output yang diharapkan:

```
ARIF-Net -- Pipeline Status

Tahap  Nama                                    Output                        Status
----------------------------------------------------------------------------------------------------
1a     Collect PIHPS (harga komoditas)         data/raw/pihps/               -- belum
1b     Collect News (berita + isi)             data/raw/news/                -- belum
...
```

---

## Penggunaan

### Quick Start

```bash
# Lihat status pipeline
python run.py status

# Jalankan collection data cepat (BBM, kalender)
python run.py collect --step bbm calendar

# Jalankan collection iklim (butuh internet, ~5 menit)
python run.py collect --step climate

# Jalankan collection harga PIHPS (butuh internet, ~30 menit)
python run.py collect --step pihps

# Jalankan collection berita (butuh internet, bisa lama)
python run.py collect --step news_idx --max-minutes 60
python run.py collect --step news_art --max-minutes 60
python run.py collect --step news_norm

# Preprocessing seluruh data
python run.py preprocess

# Training: search (default paper + grid, F1-F2) -> validate (5 seed) -> test sekali buka
python run.py train search   --model all
python run.py train validate --model all
python run.py train test     --model all --dry-run   # gladi resik di F2
python run.py train test     --model all             # SEKALI

# Uji coba satu model
python run.py train run --model gru --fold F1 --set train.epochs=1

# Gap analysis (konfigurasi beku saja)
python run.py analyze
```

### Menjalankan per Modul

Setiap modul juga bisa dijalankan langsung:

```bash
# Collection
python -m 01_collection.collect_pihps smoke        # Uji koneksi PIHPS
python -m 01_collection.collect_pihps full          # Full collection
python -m 01_collection.collect_news index          # Scrape indeks berita
python -m 01_collection.collect_news articles       # Scrape isi berita
python -m 01_collection.collect_climate             # Data iklim Open-Meteo

# Preprocessing
python -m 02_preprocessing.preprocess_pihps         # Parse harga
python -m 02_preprocessing.preprocess_news          # Sentimen berita
python -m 02_preprocessing.build_target             # Bangun target r(t,h)
python -m 02_preprocessing.build_dataset            # Dataset final

# Training
python -m 03_training.training run --model xgb --fold F1 --seed 42
python -m 03_training.analyze_gaps                  # Gap analysis
```

---

## Struktur Direktori

```
ARIF-Net/
|-- run.py                          # Entry point pipeline
|-- README.md                       # Dokumentasi utama
|-- LICENSE                         # MIT License
|-- environment.yml                 # Conda environment
|-- requirements.txt                # Python dependencies
|
|-- configs/                        # Konfigurasi
|   |-- project.json                # Global: periode, komoditas, horizon
|   |-- news_sources.json           # 3 media + URL templates
|   |-- news_scope.json             # Kata kunci filter relevansi
|   |-- locations.json              # 18 kabupaten pemasok (lat/lon)
|   +-- models/                     # Config per model (13 file)
|
|-- 01_collection/                  # Data Collection
|   |-- collect_pihps.py            # Harga komoditas (PIHPS BI)
|   |-- collect_news.py             # Berita + isi (trafilatura)
|   |-- collect_climate.py          # Iklim (Open-Meteo)
|   |-- collect_bbm.py              # Harga BBM
|   |-- collect_calendar.py         # Kalender hari libur
|   +-- utils.py                    # HTTP client, keyword matcher
|
|-- 02_preprocessing/               # Preprocessing
|   |-- preprocess_pihps.py         # Normalisasi harga
|   |-- preprocess_news.py          # Sentimen + agregasi
|   |-- preprocess_climate.py       # Normalisasi iklim
|   |-- build_target.py             # Target r(t,h)
|   |-- build_dataset.py            # Dataset multimodal final
|   +-- utils.py                    # Utilitas preprocessing
|
|-- 03_training/                    # Training & Evaluation
|   |-- training.py                 # Unified training pipeline
|   |-- metrics.py                  # MAE, RMSE, relMAE, DM test
|   |-- analyze_gaps.py             # Gap analysis
|   +-- models/                     # Implementasi model
|       |-- persistence.py          # Naive baseline
|       |-- arima.py                # ARIMA univariat
|       |-- tabular.py              # Ridge, SVR, RF, XGB, LGBM
|       |-- neural.py               # LSTM, GRU, BiLSTM, DIA-LSTM, TFT, TCN
|       +-- twostage.py             # TCN-XGBoost hybrid
|
|-- 04_arif_net/                    # ARIF-Net (setelah gap analysis)
|
|-- data/                           # Data (generated, not in git)
|   |-- raw/                        # Data mentah dari collection
|   |-- processed/                  # Data hasil preprocessing
|   +-- final/                      # Dataset final siap model
|
|-- results/                        # Hasil (generated, not in git)
|   |-- runs/                       # Output training
|   |-- evaluation/                 # Perbandingan model
|   +-- gap_analysis/               # Analisis gap
|
+-- docs/                           # Dokumentasi tambahan
    |-- SETUP.md                    # Panduan setup detail
    |-- PIPELINE.md                 # Penjelasan pipeline
    +-- DATA_DICTIONARY.md          # Kamus data
```

---

## Sumber Data

| Sumber | Deskripsi | Metode | Frekuensi |
|--------|-----------|--------|-----------|
| [PIHPS BI](https://www.bi.go.id/hargapangan) | Harga komoditas eceran | REST API | Harian (hari kerja) |
| [Kompas](https://indeks.kompas.com) | Berita headline + isi | Web scraping + trafilatura | Harian |
| [Detik Finance](https://finance.detik.com) | Berita headline + isi | Web scraping + trafilatura | Harian |
| [CNN Indonesia](https://www.cnnindonesia.com) | Berita headline + isi | Web scraping + trafilatura | Harian |
| [Open-Meteo](https://open-meteo.com) | Data iklim 18 kabupaten | REST API (gratis) | Harian |
| Kementerian ESDM | Harga BBM | Data historis (manual) | Per perubahan kebijakan |
| Keppres | Kalender hari libur | Data historis (manual) | Tahunan |

**Periode data**: 1 Januari 2019 -- 30 September 2026

---

## Model Baseline

13 model dari literatur terdahulu, dilatih dengan **aturan yang sama** (dataset, split, metrik):

| # | Model | Tipe | Referensi |
|---|-------|------|-----------|
| 1 | Persistence | Statistik | Naive baseline (prediksi r=0) |
| 2 | ARIMA | Statistik | Box-Jenkins (1970) |
| 3 | Ridge | ML Tabular | Hoerl & Kennard (1970) |
| 4 | SVR | ML Tabular | Vapnik (1995) |
| 5 | Random Forest | ML Tabular | Breiman (2001) |
| 6 | XGBoost | ML Tabular | Chen & Guestrin (2016) |
| 7 | LightGBM | ML Tabular | Ke et al. (2017) |
| 8 | LSTM | Deep Learning | Manogna et al. (2025) |
| 9 | GRU | Deep Learning | Manogna et al. (2025) |
| 10 | BiLSTM | Deep Learning | Baseline multimodal |
| 11 | DIA-LSTM | Deep Learning | Gu et al. (2022) |
| 12 | TFT | Deep Learning | Amalia et al. (2026) |
| 13 | TCN-XGBoost | Hybrid | Zhao et al. (2025) |

---

## Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **MAE** | Mean Absolute Error pada target r(t,h) |
| **RMSE** | Root Mean Square Error |
| **relMAE** | MAE / MAE_persistence (< 1.0 = lebih baik dari baseline) |
| **DA** | Directional Accuracy (arah naik/turun) |
| **MAPE** | Mean Absolute Percentage Error pada harga rekonstruksi |
| **R-squared** | Koefisien determinasi |
| **DM Test** | Diebold-Mariano test (signifikansi statistik) |

---

## Dokumentasi Lanjutan

- [Panduan Setup Detail](docs/SETUP.md) -- Instalasi langkah demi langkah
- [Penjelasan Pipeline](docs/PIPELINE.md) -- Cara kerja setiap tahap
- [Kamus Data](docs/DATA_DICTIONARY.md) -- Deskripsi fitur dan variabel

---

## Kontribusi

Proyek ini dikembangkan sebagai penelitian akademik. Untuk pertanyaan atau kontribusi:

1. Buat [Issue](https://github.com/ariftsx/ARIF-Net/issues) untuk laporan bug atau saran
2. Fork repository dan buat Pull Request untuk kontribusi kode

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

**Peneliti**: Muhamad Nur Arif  
**Program Studi**: Teknik Informatika  
**Universitas**: Universitas Mercu Buana  
**Tahun**: 2026
