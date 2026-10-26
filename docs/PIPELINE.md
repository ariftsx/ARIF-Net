# Penjelasan Pipeline

Dokumen ini menjelaskan cara kerja setiap tahap dalam pipeline ARIF-Net.

---

## Gambaran Umum

Pipeline ARIF-Net terdiri dari 4 tahap utama yang dijalankan secara berurutan:

```
Tahap 1: Collection    Mengumpulkan data mentah dari berbagai sumber
Tahap 2: Preprocessing Membersihkan, mentransformasi, dan menggabungkan data
Tahap 3: Training      Melatih dan mengevaluasi model baseline
Tahap 4: ARIF-Net      Merancang model baru berdasarkan gap analysis
```

---

## Tahap 1: Data Collection (`01_collection/`)

### 1a. PIHPS — Harga Komoditas (`collect_pihps.py`)

- **Sumber**: Bank Indonesia (bi.go.id/hargapangan)
- **Data**: Harga eceran harian 3 komoditas di Pasar Kramatjati
- **Metode**: REST API dengan session cookie (warm-up)
- **Output**: `data/raw/pihps/{comcat_id}_{YYYY-MM}.json`
- **Resume**: Otomatis melanjutkan dari chunk terakhir

### 1b. News — Berita (`collect_news.py`)

Pipeline berita terdiri dari 3 sub-tahap:

1. **Index** (`collect_news index`): Scrape halaman indeks per (media, tanggal) untuk mendapatkan daftar URL + judul artikel
2. **Articles** (`collect_news articles`): Untuk artikel yang judulnya **relevan** (keyword matching), scrape isi lengkap menggunakan **trafilatura**
3. **Normalize** (`collect_news normalize`): Gabungkan metadata + body text menjadi `articles_relevant.csv`

**Filter relevansi**: Sebelum scraping isi artikel (hemat bandwidth), judul dicocokkan dengan kata kunci di `configs/news_scope.json`. Topik meliputi komoditas, pasar/pasokan, harga/inflasi, cuaca, dan BBM/logistik.

### 1c. Climate — Iklim (`collect_climate.py`)

- **Sumber**: Open-Meteo Historical Weather API (gratis)
- **Data**: 8 variabel cuaca harian (suhu, curah hujan, kelembapan, dll.)
- **Cakupan**: 18 kabupaten sentra produksi
- **Output**: `data/raw/climate/{location_id}_{start}_{end}.json`

### 1d. BBM (`collect_bbm.py`)

- **Sumber**: SK Menteri ESDM (data historis terverifikasi)
- **Data**: Harga harian Pertalite, Solar, Pertamax
- **Metode**: Ekspansi tanggal berlaku ke harga harian

### 1e. Calendar (`collect_calendar.py`)

- **Sumber**: Keputusan Presiden tentang Hari Libur Nasional
- **Data**: Flag per hari (weekend, holiday, Ramadan, Idul Fitri, dll.)

---

## Tahap 2: Preprocessing (`02_preprocessing/`)

### 2a. Preprocess PIHPS (`preprocess_pihps.py`)

Konversi raw JSON PIHPS menjadi CSV harga harian per komoditas:
- Parse row pasar Kramatjati dari response JSON
- Filter hari kerja, hapus duplikat
- Output: `data/processed/pihps/prices_{comcat_id}.csv`

### 2b. Preprocess News (`preprocess_news.py`)

Hitung fitur berita per hari:
- **Sentimen**: Keyword-based (default) atau transformer-based (opsional)
- **Tanggal tersedia** (Data Contract §6, P1-DG-34): artikel ber-jam → tanggal terbit WIB; artikel bertanggal saja
  (tanpa jam / `T00:00:00`, mis. detik) → **+1 hari**; tanpa waktu terbit → dikeluarkan
- **Agregasi** per tanggal tersedia: jumlah berita, sentimen mean/min/max, count per topik
- Output: `data/processed/news/news_features.csv`

### 2c. Preprocess Climate (`preprocess_climate.py`)

Parse Open-Meteo JSON menjadi fitur iklim:
- Gabungkan semua lokasi
- Hitung rata-rata Tier-1 (pemasok utama)
- Output: `data/processed/climate/climate_tier1_mean.csv`

### 2d. Build Target (`build_target.py`)

Bangun target prediksi:
```
r(t,h) = (P(t+h) - P(t)) / P(t)
```
- **t**: tanggal origin (observasi harga)
- **h**: horizon (1, 3, 7, 14, 30, 90, 180, 365 hari)
- **P(t+h)**: observasi pertama >= t+h, gap <= 7 hari
- Flag: holiday_carry, spike_reversal

### 2e. Build Dataset (`build_dataset.py`)

Gabungkan semua fitur menjadi dataset final:
- **Fitur harga** (bebas skala, hanya data ≤ t): return 1–30 observasi, posisi terhadap MA & CV 5–90, volatilitas,
  satu level `price_level` = log harga. Level mentah Rp (lag/MA) tidak dipakai karena tidak stasioner
- **Fitur iklim**: Tier-1 mean, memakai data **t − 5 hari** (jeda rilis, `configs/project.json` `availability`)
- **Fitur berita**: sentimen, count, topik (14 fitur, tanggal tersedia ≤ t)
- **Fitur BBM**: perubahan log 30/365 hari dan lama sejak penyesuaian, 3 jenis BBM (harga BBM berbentuk tangga)
- **Fitur kalender**: pada origin t (`cal_*`) dan pada tanggal target t+h (`tgt_cal_*`, known future untuk TFT)
- **Fitur horizon**: one-hot + log(h)

**Split rolling-origin + purging** (`configs/project.json` `validation`):

| Peran | Origin t | Syarat label |
|---|---|---|
| F1 train / val | < 2022-07-01 / 2022-07-01 → 2023-06-30 | < 2022-07-01 / < 2024-07-01 |
| F2 train / val | < 2023-07-01 / 2023-07-01 → 2024-06-30 | < 2023-07-01 / < 2024-07-01 |
| F3 = final train | < 2024-07-01 | < 2024-07-01 |
| Test (sekali buka) | 2024-07-01 → 2026-09-30 | ≤ 2026-09-30 |

Purging membuang baris yang labelnya melewati batas, sehingga tidak ada label latih yang "melihat" periode
evaluasi. Akibatnya val F2 tidak memiliki h = 365 (labelnya jatuh di periode test) dan h = 180 hanya sebagian.

Output: `dataset_{F1,F2}_{train,val}.csv`, `dataset_final_train.csv`, `dataset_test.csv`, `feature_registry.csv`.

---

## Tahap 3: Training & Evaluation (`03_training/`)

Alur (PROTOKOL_EKSPERIMEN codebase lama, disesuaikan): **search → validate → test sekali buka → analyze**.

```bash
python run.py train search   --model all     # default paper + grid kecil, F1-F2, seed 0 -> results/frozen/
python run.py train validate --model all     # konfigurasi beku x F1-F2 x 5 seed (neural) -> results/validation/
python run.py train test     --model all --dry-run   # gladi resik di F2 (data test tidak disentuh)
python run.py train test     --model all     # SEKALI: latih final -> test -> results/test/ (kunci OPENED.json)
python run.py analyze [--source test]        # gap analysis dari konfigurasi beku saja
python run.py train run --model gru --fold F1 --set train.epochs=1   # uji coba satu model
```

| Modul | Isi |
|---|---|
| `data.py` | satu pintu data: baca split (cek purging), imputasi & standardisasi (z dipotong ±5) di-fit di LATIH per komoditas, sequence harga (L × 4: rel_logp, ret, gap, mask) dan sequence iklim t−5 (DIA-LSTM), σ_{c,h}, ambang shock Q95 |
| `trainer.py` | training loop neural (Adam; loss MSE/Huber/MAE); jumlah epoch = hyperparameter dari kurva checkpoint |
| `models/` | persistence, ARIMA (ADF → d, AIC → p,q, filter maju per origin), tabular (Ridge/SVR/RF/XGB/LGBM), neural (LSTM/GRU, BiLSTM, DIA-LSTM, TFT, TCN), TCN-XGBoost |
| `runs.py` | jalankan model × peran data, catat `results/runs/<run_id>/` |
| `search.py`, `final_test.py` | pencarian & pembekuan; validasi multi-seed; uji akhir dengan kunci |
| `metrics.py` | relMAE (seleksi), MAE, RMSE, MASE, R², DA, MAPE/MAE Rp, shock MAE (Q95 latih), DM-HLN (loss absolut) |

Kesetiaan terhadap paper dan adaptasinya: **`docs/MODEL_ADAPTATIONS.md`**.

**Komputasi.** `train.threads = "auto"` (configs/models) dan `n_jobs` model pohon memakai jumlah **core fisik**
mesin yang menjalankan; untuk menimpa: `set ARIFNET_THREADS=8` (Windows CMD) sebelum `python run.py train ...`.
Default paper LSTM/GRU (128 unit, L = 250, batch 64, sampai 50 epoch) berat: di laptop 2 core ±7,8 menit/epoch
(LSTM) dan ±3,7 menit/epoch (GRU) pada F1. Pencarian dapat dilanjutkan bila terputus (`results/search/<model>.partial.csv`),
jadi jalankan model berat terpisah, mis. `python run.py train search --model lstm`.

### Gap Analysis (`analyze_gaps.py`)

Hanya membaca hasil **konfigurasi beku** (run validasi bertag `valid`, atau prediksi test), tidak pernah run
kandidat pencarian. Menghasilkan ranking per horizon & komoditas, skenario di mana semua model gagal (gap),
perbandingan multimodal vs harga saja, dan rekomendasi arsitektur ARIF-Net.

Output: `results/gap_analysis/gap_analysis_report.md`

---

## Tahap 4: ARIF-Net (`04_arif_net/`)

Tahap ini akan diisi **setelah** gap analysis selesai.
Arsitektur model ARIF-Net dirancang berdasarkan temuan empiris dari tahap 3.
