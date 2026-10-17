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
- **Agregasi**: jumlah berita, sentimen mean/min/max, count per topik
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
- **Fitur harga**: lag, return, MA, volatilitas, momentum (~40 fitur)
- **Fitur iklim**: suhu, hujan, kelembapan, dll. (~8 fitur)
- **Fitur berita**: sentimen, count, topik (~15 fitur)
- **Fitur BBM**: harga 3 jenis BBM (~3 fitur)
- **Fitur kalender**: holiday, weekend, Ramadan, dll. (~10 fitur)
- **Fitur horizon**: one-hot + log(h)
- Split kronologis: train/val per fold + test

---

## Tahap 3: Training & Evaluation (`03_training/`)

### Training (`training.py`)

Semua model dilatih dengan aturan identik:
- Dataset yang sama (output build_dataset)
- Split kronologis yang sama (walk-forward 3-fold)
- Metrik evaluasi yang sama
- Per komoditas, pooled over horizons

### Metrics (`metrics.py`)

| Metrik | Rumus | Interpretasi |
|--------|-------|--------------|
| MAE | mean(\|y - y_hat\|) | Error rata-rata absolut |
| RMSE | sqrt(mean((y - y_hat)^2)) | Error rata-rata kuadrat |
| relMAE | MAE / MAE_persistence | < 1.0 = lebih baik dari baseline |
| DA | P(sign(y_hat) == sign(y)) | Akurasi arah perubahan |
| DM Test | Diebold-Mariano statistic | Signifikansi perbedaan model |

### Gap Analysis (`analyze_gaps.py`)

Analisis otomatis yang menghasilkan:
1. Ranking model per horizon dan komoditas
2. Skenario di mana **semua** model gagal (gap)
3. Perbandingan multimodal vs harga saja
4. Rekomendasi arsitektur ARIF-Net

Output: `results/gap_analysis/gap_analysis_report.md`

---

## Tahap 4: ARIF-Net (`04_arif_net/`)

Tahap ini akan diisi **setelah** gap analysis selesai.
Arsitektur model ARIF-Net dirancang berdasarkan temuan empiris dari tahap 3.
