# Contributing to ARIF-Net

Terima kasih atas minat Anda untuk berkontribusi pada proyek **ARIF-Net (Agricultural Risk and Inflation Forecasting Network)**! Dokumen ini memberikan panduan tata kelola dan alur kerja untuk tim internal maupun kolaborator.

---

## 📌 Alur Kerja Git & Percabangan (Branching Workflow)

1. **Branch Utama**:
   - `main`: Branch produksi/publik yang stabil dan teruji.
   - `develop` (opsional untuk fitur besar): Integrasi sebelum rilis.

2. **Branch Fitur / Eksperimen**:
   - Buat branch baru dari `main` dengan format penamaan standar:
     - `feat/<nama-fitur>`: Penambahan modul atau kemampuan baru.
     - `fix/<nama-bug>`: Perbaikan kesalahan logika atau skrip.
     - `experiment/<nama-model>`: Eksperimen model atau arsitektur baru.
     - `docs/<topik>`: Pembaruan atau penambahan dokumentasi.

---

## 📝 Format Pesan Commit (Conventional Commits)

Proyek ini menggunakan standar **Conventional Commits**:

```text
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Type yang Didukung:
- `feat`: Fitur baru (misal: penambahan model arsitektur, scraper baru).
- `fix`: Perbaikan bug atau penanganan kesalahan logika.
- `docs`: Perubahan dokumentasi (README, docs/, komentar API).
- `style`: Format kode, whitespace, semicolon tanpa perubahan logika.
- `refactor`: Restrukturisasi kode tanpa menambah fitur atau memperbaiki bug.
- `perf`: Optimasi performa atau efisiensi pipeline.
- `test`: Penambahan atau pembaruan unit/integration test.
- `chore`: Tugas pemeliharaan, build tools, dependency, .gitignore.

### Scope yang Umum:
- `collection`: Sub-modul `01_collection` (pihps, news, climate, bbm, calendar).
- `preprocessing`: Sub-modul `02_preprocessing` (fitur, target builder, normalisasi).
- `training`: Sub-modul `03_training` (model, metrik, gap analysis).
- `arif-net`: Sub-modul `04_arif_net` (arsitektur multimodal baru).
- `config`: Konfigurasi JSON di `configs/`.
- `runner`: Orkestrator pipeline `run.py`.

---

## 💻 Standar Kode & Kualitas

1. **Python Version**: Python 3.12+
2. **Gaya Kode**:
   - Mengikuti panduan **PEP 8**.
   - Gunakan Type Hinting pada fungsi dan metode baru.
   - Sertakan docstrings deskriptif pada setiap modul dan class.
3. **Encoding & Cross-Platform**:
   - Pastikan skrip kompatibel dengan Windows dan Linux.
   - Hindari karakter non-ASCII di log stdout jika berpotensi menyebabkan issue cp1252 di terminal Windows.
4. **Data Hygiene**:
   - **JANGAN PERNAH** melakukan commit data mentah berukuran besar (`data/raw/`, `data/processed/`, `data/final/`) atau checkpoint bobot model ke git. Pastikan file tersaring lewat `.gitignore`.

---

## 🚀 Prosedur Pull Request

1. Pastikan skrip berjalan tanpa error:
   ```bash
   python run.py status
   ```
2. Pastikan file konfigurasi valid JSON.
3. Buat Pull Request dengan ringkasan perubahan yang jelas, deskripsi pengujian, serta output metrik jika relevan.
