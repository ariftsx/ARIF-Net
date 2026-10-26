# DEVLOG ARIF-Net

Catatan setiap keputusan desain sesudah spesifikasi `docs/ARIFNET_SPEC.md v1.0.0` dikunci (SPEC §4). Isi setiap entri: tanggal, perubahan, alasan, skor validasi F1–F2 sebelum/sesudah. **Test lama dan data sesudah 2026-09-30 tidak boleh menjadi alasan perubahan.**

| Tanggal | Perubahan | Alasan | Skor F1–F2 sebelum → sesudah |
|---|---|---|---|
| 2026-10-27 | Implementasi awal I-1 s.d. I-6 sesuai SPEC v1.0.0 | — | — |
| 2026-10-27 | Catatan (bukan perubahan): di data LATIH F1, korelasi d_trend·h dengan r bernilai negatif (h90 −0,39, h180 −0,54, h365 −0,34) | Pola mean-reversion pada rezim latih. λ ∈ [0, 1] (SPEC §2.4) membuat model dapat mematikan drift bila tidak membantu; diamati lewat `lambda_drift` di result.json | — |
