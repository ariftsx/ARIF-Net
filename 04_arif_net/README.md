# ARIF-Net v1 — Model Usulan (Tahap 4)

Spesifikasi: [`docs/ARIFNET_SPEC.md`](../docs/ARIFNET_SPEC.md) · Protokol uji: [`docs/PROTOKOL_UJI_PROSPEKTIF.md`](../docs/PROTOKOL_UJI_PROSPEKTIF.md) (keduanya `DECIDED` 2026-10-27).

ARIF-Net dibangun dari celah benchmark 13 model:

| Celah | Komponen |
|---|---|
| G-1: zero-inflation + loss MSE ≠ metrik MAE | kepala **kejadian** P(r ≠ 0) + prediksi = **median campuran** (0 bila P ≤ 0,5) |
| G-2: besaran shock runtuh | kepala **kuantil** monoton (τ 0,1–0,9) + bobot shock |
| G-3: drift horizon panjang | komponen **drift** (Theil–Sen + musiman tahunan, hanya harga ≤ t) × λ_{c,h} |
| G-4 (ablation): heterogenitas | gerbang expert ARIF-Net / LightGBM / persistence per (c, h), bobot dari OOF ter-purge |
| G-5 (ablation): multimodal | news / iklim / BBM bergerbang |

## Berkas

```
04_arif_net/
├── model.py      ARIFNet (encoder GRU|LSTM|TCN, kepala kejadian, kuantil monoton, drift), mixture_median
├── train.py      loss BCE + pinball (+ bobot shock); data, split, preprocessing, epoch = sama dengan baseline
├── gate.py       ablation X-1 (gerbang expert)
├── selftest.py   uji unit: median campuran, monotonisitas kuantil, drift tanpa kebocoran
└── DEVLOG.md     catatan iterasi desain (wajib, SPEC §4)
configs/models/arifnet.json      A0 (default)
configs/search/arifnet.json      ruang search (≤ 40 kandidat, cek ulang top-3 × 5 seed)
configs/ablations/arifnet.json   varian C-1..C-5, E-1, M-1..M-4, X-1
```

## Menjalankan (urutan protokol)

```bash
python -m 04_arif_net.selftest                      # uji unit
python run.py preprocess                            # fitur drift + split prospektif
python run.py train search   --model arifnet        # F1–F2, lalu bekukan
python run.py train validate --model arifnet        # A0 × 5 seed
python run.py train ablate   --model arifnet        # semua varian × 5 seed
# commit kode, lalu SEBELUM data sesudah 2026-09-30 dikumpulkan:
python run.py train register --model all            # kunci: snapshot latih + hash konfigurasi/kode
# kumpulkan data sampai 2027-01-15 (project.json period.end/data_end), preprocess, lalu SEKALI:
python run.py train prospective --model all
```
