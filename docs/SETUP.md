# Panduan Setup Detail

Dokumen ini menjelaskan langkah-langkah instalasi dan konfigurasi ARIF-Net secara detail.

---

## Prasyarat

Pastikan perangkat lunak berikut sudah terinstal:

- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- **Conda** (opsional, [download Miniconda](https://docs.conda.io/en/latest/miniconda.html))

## Instalasi

### Opsi A: Menggunakan Conda (Direkomendasikan)

```bash
# 1. Clone repository
git clone https://github.com/ariftsx/ARIF-Net.git
cd ARIF-Net

# 2. Buat environment dari file konfigurasi
conda env create -f environment.yml

# 3. Aktifkan environment
conda activate ARIF-Net

# 4. Verifikasi
python run.py status
```

### Opsi B: Menggunakan pip + venv

```bash
# 1. Clone repository
git clone https://github.com/ariftsx/ARIF-Net.git
cd ARIF-Net

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan (Windows)
.venv\Scripts\activate
# Aktifkan (Linux/Mac)
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verifikasi
python run.py status
```

## Konfigurasi

### Konfigurasi Global (`configs/project.json`)

File ini berisi pengaturan utama proyek:

- **period**: Rentang waktu data (2019-01-01 s/d 2026-09-30)
- **commodities**: 3 komoditas target beserta ID PIHPS
- **horizons**: 8 horizon prediksi (1, 3, 7, 14, 30, 90, 180, 365 hari)
- **validation**: Konfigurasi fold validasi walk-forward
- **training**: Hyperparameter default (batch size, learning rate, dll.)

### Konfigurasi Berita (`configs/news_sources.json`)

Berisi URL template dan parameter scraping untuk 3 media:
- Kompas (`indeks.kompas.com`)
- Detik Finance (`finance.detik.com`)
- CNN Indonesia (`cnnindonesia.com`)

### Filter Relevansi (`configs/news_scope.json`)

Kata kunci yang digunakan untuk memfilter berita relevan **sebelum** scraping isi artikel.
Dikelompokkan per topik: komoditas, pasar/pasokan, harga/inflasi, cuaca, BBM/logistik.

### Lokasi Iklim (`configs/locations.json`)

18 kabupaten sentra produksi beserta koordinat (lat/lon) untuk query Open-Meteo API.
Dibagi menjadi Tier 1 (pemasok utama) dan Tier 2 (sekunder).

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'trafilatura'`

```bash
pip install trafilatura
```

### Error: `ModuleNotFoundError: No module named 'torch'`

PyTorch versi CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Error: `UnicodeEncodeError` pada Windows

Jalankan dengan flag UTF-8:
```bash
python -X utf8 run.py status
```

### Error: PIHPS connection timeout

Pastikan koneksi internet stabil. PIHPS BI terkadang lambat.
Gunakan mode `smoke` untuk menguji koneksi terlebih dahulu:
```bash
python -m 01_collection.collect_pihps smoke
```
