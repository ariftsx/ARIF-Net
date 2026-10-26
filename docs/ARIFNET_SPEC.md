# Spesifikasi ARIF-Net v1

**Versi:** `v1.0.0` · **Tanggal:** 2026-10-27 · **Status:** `DECIDED`. Disetujui owner pada 2026-10-27, dan tidak diubah sesudah hasil pencarian terlihat (perubahan desain hanya lewat DEVLOG dengan alasan).
**Dasar empiris:** benchmark 13 model (search F1–F2 → validasi 5 seed → test 2024-07 s.d. 2026-09, dibuka 2026-10-26).
**Aturan:** desain dan seleksi ARIF-Net **hanya** memakai F1–F2. Angka test lama hanya dipakai secara deskriptif dan bukan dasar keputusan desain. Uji akhir dilakukan secara prospektif (`docs/PROTOKOL_UJI_PROSPEKTIF.md`).

---

## 1. Celah dari benchmark (terkonfirmasi di validasi F1–F2)

| ID | Celah | Bukti validasi | Peran di ARIF-Net |
|---|---|---|---|
| **G-1** | **Zero-inflation + loss ≠ metrik.** Di horizon pendek dan pada beras, tidak ada model yang mengalahkan persistence. | 8/24 sel tanpa model < 1 (semua h=1, bawang h3, beras h3–h30). Porsi r = 0 di data latih: beras h1 95,7%, bawang h1 80,7%, cabai h1 69,7%. Bila P(r=0) > 0,5, median = 0, sehingga prediksi optimal di bawah MAE/relMAE adalah 0. Model ber-loss MSE memprediksi rata-rata dan kalah: BiLSTM beras h1 3,05×, Ridge 3,08×. TFT (loss MAE) mendapat ±1,00. | **Inti** |
| **G-2** | **Shock: besaran runtuh.** | Pada \|r\| ≥ Q95 latih, \|r̂\|/\|r\| hanya 0,02–0,20. relMAE shock semua model 0,93–0,99. | **Inti** |
| **G-3** | **Drift horizon panjang salah atau terlewat.** | Rata-rata r beras F1 h365 = +11,1%, prediksi semua model ±0. Cabai F1 h365 +27,1%, sedangkan SVR −9,7%, Ridge −13,0%, BiLSTM −3,3% (*mean-reversion* dari rezim latih). | **Inti** |
| **G-4** | **Heterogenitas: tidak ada pemenang di semua sel.** | Pemenang per sel berbeda (TFT, BiLSTM, LightGBM, RF, LSTM, XGB). Korelasi galat TFT dengan model lain 0,72–0,85. Rata-rata top-3 per sel hasil seleksi validasi: F1→F2 0,841 (BiLSTM 0,892), F2→F1 0,879 (BiLSTM 0,831), sehingga **campuran**. | **Ablation** |
| **G-5** | **Multimodal lewat concat tidak membantu.** | Ridge harga 0,874 vs multimodal 1,079; SVR 0,897 vs 0,959; LightGBM 0,887 vs 0,881; TCN-XGB tanpa beda. Eksperimen lama: gerbang modality tidak bereaksi terhadap reliabilitas (K-4 gagal). | **Ablation** |

Batasan data: val F2 tidak memiliki h=365 karena purging, sehingga keputusan untuk h=365 hanya bertumpu pada F1.

---

## 2. Arsitektur

```
 sequence harga (L×4) ─► Encoder temporal (GRU) ─┐
 fitur harga + kalender origin/target ───────────┼─► z  (representasi bersama, + embedding c, h)
 [G-5] news / iklim / BBM ─► proyeksi bergerbang ┘
                                                   │
                       ┌───────────────────────────┼─────────────────────────────┐
                       ▼                           ▼                             ▼
            Kepala KEJADIAN (G-1)      Kepala BESARAN kuantil (G-2)      Komponen DRIFT (G-3)
            p = P(r ≠ 0 | z)           Q̃_τ(z), τ ∈ {.1,.25,.5,.75,.9}    d(c,h,t) deterministik dari harga ≤ t
                       │               (monoton, untuk r ≠ 0)             λ_{c,h} ∈ [0,1] dipelajari
                       │                           │                             │
                       │                           └──────────► Q_τ = λ_{c,h}·d + Q̃_τ
                       └────────────────────────────────────────────────┐
                                                                         ▼
                                Prediksi titik = MEDIAN campuran: F(x) = (1−p)·1[x ≥ 0] + p·G(x)
                                (optimal untuk galat absolut → selaras relMAE)
```

### 2.1 Encoder temporal
- GRU satu layer atas sequence harga (4 kanal, kontrak sequence yang sama dengan baseline), ditambah embedding komoditas dan horizon (adaptasi A-2).
- Dipilih karena GRU paling stabil pada beras dan horizon panjang di validasi (F2 0,857). Alternatif encoder (TCN, LSTM) diuji sebagai ablation E-1.
- Fitur harga tabular dan kalender (origin `cal_*`, target `tgt_cal_*`) masuk lewat proyeksi linear.

### 2.2 Kepala kejadian, G-1 (hurdle)
- Target biner `1[r ≠ 0]`, dengan ambang ε = 0 karena PIHPS berkelipatan Rp 50.
- Loss: binary cross-entropy.
- Input utama adalah kalender. Libur dan Ramadan memengaruhi apakah harga berubah, bukan seberapa besar.

### 2.3 Kepala besaran kuantil, G-2
- Lima kuantil bersyarat untuk r pada baris dengan r ≠ 0.
- Monotonisitas dijamin dengan parametrisasi: Q̃_.5 ditambah selisih kumulatif softplus.
- Loss: pinball (rata-rata atas τ). Opsi *shock-aware*: bobot w untuk \|r\| ≥ Q95 latih per (c, h), sama dengan A10 protokol lama. Nilai w dipilih di search.
- Target dinormalisasi dengan σ_{c,h} latih (A-3).

### 2.4 Komponen drift, G-3
- d(c,h,t) dihitung **hanya dari harga ≤ t**, tanpa parameter yang dilatih:
  - d_trend: kemiringan log-harga yang robust (Theil–Sen) atas W_d hari terakhir × h, dengan W_d ∈ {365, 730} (search).
  - d_season: rata-rata log P(t−365k+h) − log P(t−365k) untuk k = 1..K (K ≤ 3, sesuai riwayat yang tersedia).
  - d = α·d_trend + (1−α)·d_season, dengan α ∈ {0, 0,5, 1} (search).
- λ_{c,h} = sigmoid(θ_{c,h}), diinisialisasi agar λ ≈ 0. Model mulai tanpa drift dan hanya memakainya bila menurunkan loss latih.

### 2.5 Prediksi titik (median campuran)
Dengan p = P(r ≠ 0), G = CDF bersyarat r ≠ 0, dan Q = G⁻¹ (interpolasi linear antar kuantil Q_τ, diekstrapolasi datar di ujung), CDF campuran adalah F(x) = (1−p)·1[x ≥ 0] + p·G(x). Misalkan a = p·G(0⁻), yaitu massa di bawah 0:
- a ≥ 0,5 → r̂ = Q(0,5 / p) (median negatif);
- a < 0,5 ≤ a + (1−p) → **r̂ = 0**. Ini selalu terjadi bila p ≤ 0,5, dan itulah perilaku "tidak merugikan" di horizon pendek dan pada beras;
- a + (1−p) < 0,5 → r̂ = Q((p − 0,5) / p) (median positif).

Definisi di kode: `04_arif_net/model.py::mixture_median`, dengan uji unit untuk ketiga kasus.

### 2.6 Ablation G-4 (gerbang expert)
- Expert: ARIF-Net inti, LightGBM multimodal (konfigurasi beku), dan persistence.
- Bobot w_{c,h} = softmax per (c, h), di-fit pada **prediksi out-of-fold ter-purge** di dalam data latih (blocked time-series CV, embargo h hari), loss absolut.
- Dilaporkan sebagai varian `arifnet+gate`. Bukan default.

### 2.7 Ablation G-5 (modality)
- News, iklim, dan BBM masuk lewat proyeksi bergerbang ke z, dan juga ke kepala kejadian.
- Varian ablation: tanpa modality eksternal (default inti), +news, +iklim, +BBM, +semua.

---

## 3. Varian ablation (didaftarkan sebelum hasil)

| Kode | Varian | Menguji |
|---|---|---|
| **A0** | ARIF-Net inti = encoder + hurdle + kuantil + drift, modality harga + kalender | — |
| C-1 | tanpa hurdle (regresi kuantil langsung atas semua r) | G-1 |
| C-2 | tanpa drift (λ ≡ 0) | G-3 |
| C-3 | tanpa kepala kuantil, titik via MAE | G-2 |
| C-4 | loss MSE titik (kontrol ketidakcocokan loss/metrik) | G-1 |
| C-5 | tanpa bobot shock | G-2 |
| E-1 | encoder TCN / LSTM menggantikan GRU | pilihan encoder |
| M-1…M-4 | +news, +iklim, +BBM, +semua modality | G-5 |
| X-1 | `arifnet+gate` (expert gating) | G-4 |

---

## 4. Seleksi (F1–F2 saja)

- **Skor seleksi:** rata-rata relMAE F1 dan F2, sama dengan baseline. Jumlah epoch adalah hyperparameter kurva (PROTOKOL v1.0.1).
- **Anggaran:** ≤ 40 konfigurasi untuk A0, seed 0. Tiga kandidat terbaik dicek ulang dengan 5 seed, lalu yang terbaik (rata-rata 5 seed) dibekukan.
- **Ruang search:** d ∈ {32, 64, 128}, dropout ∈ {0, 0,2, 0,3}, lr ∈ {1e-4, 1e-3}, L ∈ {30, 60, 120, 250}, w_shock ∈ {1, 2}, W_d ∈ {365, 730}, α ∈ {0, 0,5, 1}.
- **Iterasi desain:** setiap perubahan arsitektur sesudah melihat skor validasi dicatat di `04_arif_net/DEVLOG.md` (tanggal, alasan, skor sebelum/sesudah).
- **Larangan:** test lama (2024-07 → 2026-09) tidak boleh dipakai untuk memilih atau menyetel apa pun.

---

## 5. Kriteria klaim (didaftarkan sebelum hasil)

Ditetapkan di `docs/PROTOKOL_UJI_PROSPEKTIF.md` §5 (K-1 s.d. K-5).

---

## 6. Rencana implementasi

| Langkah | Berkas | Isi |
|---|---|---|
| I-1 | `02_preprocessing/build_dataset.py` | fitur drift d_trend dan d_season per (c, origin, h), dari harga ≤ t |
| I-2 | `04_arif_net/model.py` | encoder, kepala kejadian, kepala kuantil monoton, λ_{c,h}, `mixture_median` + uji unit |
| I-3 | `04_arif_net/train.py` | loss gabungan BCE + pinball (+ bobot shock), memakai `03_training/data.py` (data, split, dan preprocessing identik dengan baseline) |
| I-4 | `03_training/runs.py` | family `arifnet` agar search, validate, dan uji prospektif memakai jalur yang sama dengan baseline |
| I-5 | `configs/models/arifnet*.json`, `configs/search/arifnet.json` | konfigurasi A0 dan varian ablation |
| I-6 | `03_training/final_test.py` | peran `prospektif` dengan kunci terpisah (protokol §4) |
