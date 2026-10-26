# Kesetiaan Model Pembanding terhadap Paper & Adaptasinya

**Kebijakan (keputusan peneliti, 2026-10-23):** *paper default + search*. Hyperparameter paper asli menjadi
konfigurasi default (`configs/models/<model>.json`, kandidat 0 pencarian); grid kecil di `configs/search/<model>.json`
menyertakan nilai paper. Pemilihan memakai rata-rata relMAE F1–F2 (data test tidak dipakai).

## Adaptasi bersama (berlaku sama untuk semua model, wajib karena desain penelitian)

| # | Adaptasi | Alasan |
|---|---|---|
| A-1 | Target = return r(t,h) = (P(t+h) − P(t)) / P(t), 8 horizon (1, 3, 7, 14, 30, 90, 180, 365 hari), *direct multi-horizon* | Paper meramal harga level 1 langkah / multi-step; ARIF-Net membandingkan semua model pada target dan horizon yang sama |
| A-2 | Model neural *pooled* 3 komoditas × 8 horizon dengan embedding komoditas & horizon; lapis keluaran diinisialisasi nol (epoch 0 = persistence) | Satu model per (c, h) = 24 model per paper → data per model terlalu sedikit; perlakuan identik untuk semua model neural |
| A-3 | Target distandardisasi σ_{c,h} data latih (pengganti MinMax harga di paper) | Target sudah berupa return; skala antar horizon sangat berbeda |
| A-4 | Jumlah epoch = hyperparameter dari kurva checkpoint F1–F2 (pengganti early stopping 20% data latih) | PROTOKOL v1.0.1: sub-validasi di dalam data latih hampir hanya berisi horizon pendek karena purging |
| A-5 | Split rolling-origin F1–F2 + purging label; uji akhir sekali buka | Paper memakai satu split 80/20 atau 70/15/15 tanpa purging (berisiko bocor pada target horizon panjang) |
| A-6 | 5 seed (0–4) untuk konfigurasi beku, prediksi test = rata-rata seed | PROTOKOL P4-DG-07 |

## Per model

| Model | Paper | Default di codebase | Adaptasi khusus |
|---|---|---|---|
| ARIMA | Manogna et al. (2025) Alg. ARIMA: ADF → differencing, ordo via AIC/BIC; Amalia et al. (2026): `auto_arima` | log harga hari terlapor; d dari ADF (α 0,05); (p, q) ∈ {0,1,2,3,5} × {0,1,2} via AIC; fit sekali per fold, filter maju per origin | Langkah = jumlah hari kerja di (t, t+h] (PROTOKOL P4-DG-09) |
| SVR | Manogna et al. (2025): kernel RBF, C, ε, harga dinormalisasi | RBF, C 1, ε 0,1, fitur terstandardisasi | Fitur multimodal/harga (pilihan search) |
| LSTM / GRU | Manogna et al. (2025): 1 layer **128 unit**, dropout **0,2**, Dense(1) linear, **MSE**, **Adam**, batch **64**, **50 epoch**, univariat, window **1000** | d 128, dropout 0,2, MSE, Adam 1e-3, batch 64, maks 50 epoch, input sequence harga saja, **L = 250** | Window 1000 tidak mungkin (±1.950 observasi per komoditas, fold F1 ±850) → L ∈ {30, 60, 120, 250} |
| DIA-LSTM | Gu et al. (2022) Gambar 2–3b, Tabel 2: atensi fitur (linear + softmax atas fitur) dan atensi temporal (linear + softmax atas waktu pada Xᵀ), keduanya dikalikan ke input lalu di-concat → **LSTM 6 unit** (tanh, stateful) → **dropout 0,2** → **FC 10** → FC 1; **Adam 1e-3**, **MSE**; time step **T = 6** (grid {1, 2, 4, 6, 8, 12}); input harga + volume + meteorologi | Persis struktur tersebut; input = sequence harga (4 kanal) + iklim Tier-1 (8 variabel, t−5) + mask; T default 6 | Data bulanan → harian (T search {4, 6, 8, 12, 30, 60}); tanpa volume perdagangan (tidak tersedia); stateful batch 1 → batch 64 non-stateful (sampel pooled tidak berurutan) |
| TFT | Amalia et al. (2026): Darts TFT, harga saja, quantile forecast, tuning Optuna: hidden {16…128}, dropout {0…0,3}, lr {1e-5…1e-2}, heads {1…4}; Lim et al. (2021): GRN, variable selection, LSTM encoder, interpretable multi-head attention, static & known future inputs | GRN, variable selection, LSTM encoder dengan state awal dari static (c, h), interpretable MHA, known future = kalender tanggal target; d 32, heads 4, dropout 0,1, loss **MAE (= kuantil 0,5)** | Point forecast median (bukan seluruh kuantil); satu langkah r(t,h) per baris (bukan decoder 30 langkah) |
| TCN-XGBoost | Zhao et al. (2025): TCN **6 layer**, kernel **3**, dilasi **1…32**, dropout **0,3**, **WeightNorm**, window **30**; fitur TCN → XGBoost **max_depth 10**, **lr 0,03**, subsample 1,0, λ 1,2, loss kuadrat | Persis; TCN dilatih end-to-end (MSE) lalu representasinya → XGBoost per komoditas | Epoch TCN = hyperparameter search {5, 10, 20, 50}; opsi `stage2.features = multimodal` (varian protokol lama) di search |
| Ridge, RF, XGBoost, LightGBM | Baseline umum (bukan replikasi paper tertentu) | Per komoditas, pooled horizon; Ridge/SVR pada fitur terstandardisasi | Varian fitur harga vs multimodal dipilih search |
| BiLSTM | Baseline deep (concat semua modality) | BiLSTM harga + proyeksi fitur tabular multimodal | — |

## Hal yang belum/tidak bisa direplikasi

- **Volume perdagangan** (input DIA-LSTM) tidak tersedia untuk pasar Kramatjati.
- **Sentimen berita** memakai skor berbasis kata kunci (`preprocess_news.py`); opsi model transformer
  (`--use-model`) belum dipakai di pipeline default.
- **Angka di paper asli tidak dibandingkan langsung** dengan hasil di sini (data, target, dan horizon berbeda).
