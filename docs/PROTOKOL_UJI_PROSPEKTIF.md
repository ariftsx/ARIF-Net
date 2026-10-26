# Protokol Uji Prospektif ARIF-Net

**Versi:** `v1.0.0` · **Tanggal:** 2026-10-27 · **Status:** `DECIDED`. Disetujui owner pada 2026-10-27, sebelum data PIHPS sesudah 2026-09-30 dikumpulkan atau dilihat.
**Keputusan owner (2026-10-27):** desain dan seleksi ARIF-Net hanya di F1–F2, lalu uji buta pada data PIHPS sesudah 2026-09-30 bersama semua baseline beku.

---

## 1. Alasan

Test 2024-07-01 → 2026-09-30 sudah dibuka pada 2026-10-26 dan hasil baseline sudah diketahui peneliti. Model yang dirancang sesudah itu tidak bisa dinilai secara buta pada periode yang sama. Uji prospektif memakai data yang **belum ada atau belum dilihat** saat semua model dibekukan, sehingga menjadi satu-satunya uji konfirmatori untuk klaim "ARIF-Net mengungguli".

## 2. Peran data

| Peran | Origin t | Label | Dipakai untuk |
|---|---|---|---|
| F1, F2 | sama dengan `configs/project.json` | dengan purging | desain, seleksi, ablation (validasi 5 seed) |
| Test lama | 2024-07-01 → 2026-09-30 | ≤ 2026-09-30 | **eksploratif saja**, dilaporkan terpisah dan diberi label |
| **Latih prospektif** | ≤ 2026-09-30 | ≤ 2026-09-30 | fit ulang SEMUA model (ARIF-Net + 13 baseline) dengan konfigurasi beku |
| **Uji prospektif** | 2026-10-01 → C | ≤ C | uji buta, sekali buka |

**C = tanggal cutoff = `2027-01-15`** (keputusan owner, 2026-10-27).

Dengan C = 2027-01-15, perkiraan jumlah origin per komoditas (hari kerja, belum dikurangi libur):

| h | 1 | 3 | 7 | 14 | 30 | 90 | 180 | 365 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| origin | 76 | 74 | 72 | 67 | 55 | 12 | 0 | 0 |

**Konsekuensi:** uji prospektif hanya bisa mengonfirmasi **h ∈ {1, 3, 7, 14, 30}**. h = 90 dilaporkan indikatif (12 origin, sekitar 0,1 jendela independen). h = 180 dan 365 **tidak dapat diuji secara prospektif sebelum sidang**: klaim untuk horizon panjang (G-3) hanya bertumpu pada validasi F1–F2, dan dinyatakan sebagai keterbatasan.

## 3. Urutan kerja (tidak boleh dibalik)

1. **Kunci protokol & spesifikasi**: owner menyetujui dokumen ini dan `docs/ARIFNET_SPEC.md`, lalu status menjadi `DECIDED`.
2. **Desain & seleksi ARIF-Net** di F1–F2 (SPEC §4). Iterasi dicatat di `04_arif_net/DEVLOG.md`.
3. **Pembekuan**: `results/frozen/arifnet*.json` (A0 + varian ablation), dengan baseline tetap memakai konfigurasi beku 2026-10-26.
4. **Registrasi**: `results/prospektif/REGISTERED.json` ditulis **sebelum** pengumpulan data sesudah 2026-09-30. Isinya:
   - sha256 semua konfigurasi beku;
   - commit git kode;
   - sha256 `dataset_final` / latih prospektif;
   - nilai C;
   - daftar model;
   - kriteria §5.
5. **Fit ulang** semua model pada peran latih prospektif (5 seed untuk model neural, ensembel rata-rata seed).
6. **Pengumpulan data** PIHPS, berita, iklim, BBM, dan kalender sampai C dengan collector yang sama (tanpa perubahan kode), lalu preprocessing dengan kode yang sama.
7. **Pembukaan sekali**: `run.py train test --prospektif` memverifikasi kecocokan hash dengan REGISTERED.json, menulis `OPENED.json`, lalu memprediksi dan menilai. Pembukaan kedua ditolak.
8. **Laporan**: hasil dilaporkan apa adanya, termasuk kriteria yang gagal.

## 4. Aturan anti-kebocoran

- **Kode beku:** tidak ada perubahan pada kode model, fitur, atau preprocessing sesudah registrasi, kecuali perbaikan collector yang tidak mengubah nilai data. Perbaikan seperti itu dicatat dan hash data diverifikasi.
- **Data ≤ t:** aturan ketersediaan waktu tetap sama (iklim t−5, berita tanggal tersedia, harga ≤ t).
- **Tanpa pengecekan sebelum pembukaan:** prediksi atau angka uji prospektif tidak dilihat sebelum langkah 7.
- **Pipeline otomatis:** pipeline mingguan (Airflow/PC) **tidak** menjalankan training atau evaluasi pada data sesudah 2026-09-30 sebelum langkah 7.

## 5. Kriteria klaim (didaftarkan sebelum hasil)

Semua angka di bawah memakai uji prospektif pada h ∈ {1, 3, 7, 14, 30} kecuali disebut lain. Model neural dinilai dengan ensembel 5 seed.

| # | Kriteria | Ukuran | Status bila gagal |
|---|---|---|---|
| **K-1** | ARIF-Net (A0) lebih akurat daripada **semua** 13 baseline | relMAE gabungan uji prospektif terendah | klaim utama gugur |
| **K-2** | Keunggulan atas baseline terkuat **signifikan** | DM-HLN (loss absolut, HAC lag k−1) per (komoditas, horizon) vs **BiLSTM** (skor validasi terbaik) dan vs **SVR** (terbaik di test lama): ARIF-Net lebih baik signifikan (p < 0,05) di lebih banyak sel daripada lebih buruk signifikan | klaim signifikansi gugur |
| **K-3** | **Tidak merugikan di horizon pendek** (G-1) | relMAE ARIF-Net ≤ 1,00 untuk setiap komoditas pada h ∈ {1, 3, 7} | — |
| **K-4** | Komponen inti berkontribusi | C-1 (tanpa hurdle) dan C-2 (tanpa drift) lebih buruk daripada A0 di validasi F1–F2 (rata-rata 5 seed). C-1 juga lebih buruk di uji prospektif. | komponen dilaporkan tidak berkontribusi |
| **K-5** | Horizon panjang (G-3) | relMAE A0 pada h ∈ {90, 180, 365} di validasi F1–F2 lebih rendah daripada setiap baseline. **Dinyatakan sebagai hasil validasi, bukan uji buta.** | — |

Hasil eksploratif yang **tidak** dipakai untuk klaim: test lama, shock MAE, ablation G-4/G-5, dan h = 90 prospektif.

## 6. Pelaporan

- **Tabel utama:** relMAE, MAE, RMSE, DA, MAPE harga, dan MAE Rp per model × horizon (uji prospektif).
- **Validasi F1–F2:** mean ± std 5 seed untuk semua model dan semua ablation.
- **DM-HLN** per (komoditas, horizon).
- **Status K-1 s.d. K-5**, ditulis apa adanya.
- **Keterbatasan:** satu jendela prospektif ±3,5 bulan, horizon panjang hanya divalidasi di F1–F2, val F2 tanpa h = 365, dan test lama sudah terlihat.

## 7. Riwayat versi

| Versi | Tanggal | Perubahan |
|---|---|---|
| v1.0.0-draft | 2026-10-27 | Draf awal |
| v1.0.0 | 2026-10-27 | Disetujui owner; cutoff C = 2027-01-15 |
