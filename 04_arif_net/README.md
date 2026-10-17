# 🧠 ARIF-Net — Model Baru (Tahap 4)

> **Folder ini akan diisi setelah tahap Gap Analysis selesai.**

## Alur Pengembangan

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 01_collection   │ ──▶ │ 02_preprocessing │ ──▶ │ 03_training      │
│ (Data Mining)   │     │ (Feature Eng.)   │     │ (Baseline + Eval)│
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ Gap Analysis     │
                                                │ (Weakness Found) │
                                                └────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ 04_arif_net      │  ◀── ANDA DI SINI
                                                │ (Novel Model)    │
                                                └──────────────────┘
```

## Prasyarat

Sebelum mengembangkan arsitektur ARIF-Net:

1. ✅ Semua data sudah di-collect (01_collection)
2. ✅ Preprocessing dan dataset final sudah siap (02_preprocessing)
3. ✅ Semua model baseline & penelitian terdahulu sudah dilatih (03_training)
4. ✅ Gap analysis sudah mengidentifikasi kelemahan (03_training/analyze_gaps.py)
5. 🔲 Desain arsitektur berdasarkan temuan gap analysis
6. 🔲 Implementasi dan uji coba

## File yang Akan Dibuat

```
04_arif_net/
├── model.py         # Arsitektur ARIF-Net
├── training.py      # Training loop khusus
├── config.json      # Hyperparameter
└── README.md        # Dokumentasi arsitektur & justifikasi
```
