"""
ARIF-Net -- Gap Analysis: Analisis kelemahan model terdahulu -> rekomendasi ARIF-Net.

Membaca hasil evaluasi semua model baseline dan model penelitian terdahulu,
kemudian menganalisis:
  1. Horizon mana yang paling sulit? (h pendek vs panjang)
  2. Komoditas mana yang paling sulit?
  3. Model mana yang terbaik untuk skenario tertentu?
  4. Di mana semua model gagal? -> celah/gap
  5. Apakah fitur multimodal membantu vs hanya harga?
  6. Rekomendasi arsitektur ARIF-Net berdasarkan gap

Output: results/gap_analysis/gap_analysis_report.md
        results/gap_analysis/gap_analysis.json

Contoh:
  python -m 03_training.analyze_gaps
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import basic, diebold_mariano

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
GAP_DIR = RESULTS / "gap_analysis"

COMMODITY_NAMES = {
    "com_14": "Cabai Merah Keriting",
    "com_11": "Bawang Merah Ukuran Sedang",
    "com_3": "Beras Kualitas Medium I",
}


def load_all_results() -> pd.DataFrame:
    """Baca semua hasil training dari results/runs/."""
    rows = []
    if not RUNS.exists():
        return pd.DataFrame()
    for run_dir in sorted(RUNS.iterdir()):
        result_file = run_dir / "result.json"
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text(encoding="utf-8"))
        row = {
            "model": result.get("model_name", ""),
            "fold": result.get("fold", ""),
            "seed": result.get("seed", 0),
            "train_seconds": result.get("train_seconds", 0),
        }
        row.update(result.get("metrics_overall", {}))
        for ch in result.get("metrics_per_ch", []):
            rows.append({**row, **ch})
    return pd.DataFrame(rows)


def analyze_by_horizon(df: pd.DataFrame) -> dict:
    """Analisis performa per horizon."""
    analysis = {}
    for h in sorted(df["h"].unique()):
        h_df = df[df["h"] == h]
        best_model = h_df.loc[h_df["relmae"].idxmin(), "model"] if not h_df.empty else None
        worst_model = h_df.loc[h_df["relmae"].idxmax(), "model"] if not h_df.empty else None
        analysis[int(h)] = {
            "avg_relmae": float(h_df["relmae"].mean()),
            "best_model": best_model,
            "best_relmae": float(h_df["relmae"].min()),
            "worst_model": worst_model,
            "worst_relmae": float(h_df["relmae"].max()),
            "all_models_above_1": bool((h_df["relmae"] > 1.0).all()),
            "n_models_beating_persistence": int((h_df["relmae"] < 1.0).sum()),
        }
    return analysis


def analyze_by_commodity(df: pd.DataFrame) -> dict:
    """Analisis performa per komoditas."""
    analysis = {}
    for c in sorted(df["komoditas"].unique()):
        c_df = df[df["komoditas"] == c]
        best_model = c_df.loc[c_df["relmae"].idxmin(), "model"] if not c_df.empty else None
        analysis[c] = {
            "name": COMMODITY_NAMES.get(c, c),
            "avg_relmae": float(c_df["relmae"].mean()),
            "best_model": best_model,
            "best_relmae": float(c_df["relmae"].min()),
            "hardest_horizon": int(c_df.loc[c_df["relmae"].idxmax(), "h"]) if not c_df.empty else None,
        }
    return analysis


def find_gaps(df: pd.DataFrame) -> list[dict]:
    """Cari celah/gap: skenario di mana SEMUA model gagal."""
    gaps = []
    for (c, h), group in df.groupby(["komoditas", "h"]):
        if (group["relmae"] > 1.0).all():
            gaps.append({
                "komoditas": c,
                "commodity_name": COMMODITY_NAMES.get(c, c),
                "h": int(h),
                "best_relmae": float(group["relmae"].min()),
                "best_model": group.loc[group["relmae"].idxmin(), "model"],
                "issue": "Semua model lebih buruk dari persistence",
            })
        elif (group["relmae"] > 0.95).all():
            gaps.append({
                "komoditas": c,
                "commodity_name": COMMODITY_NAMES.get(c, c),
                "h": int(h),
                "best_relmae": float(group["relmae"].min()),
                "best_model": group.loc[group["relmae"].idxmin(), "model"],
                "issue": "Semua model hanya marginal lebih baik dari persistence",
            })
    return gaps


def multimodal_vs_price_only(df: pd.DataFrame) -> dict:
    """Bandingkan model multimodal vs hanya harga."""
    price_models = {"persistence", "arima", "lstm", "gru"}
    multimodal_models = {"ridge", "svr", "rf", "xgb", "lgbm", "bilstm", "dialstm", "tft"}

    price_df = df[df["model"].isin(price_models)]
    multi_df = df[df["model"].isin(multimodal_models)]

    return {
        "price_only_avg_relmae": float(price_df["relmae"].mean()) if not price_df.empty else None,
        "multimodal_avg_relmae": float(multi_df["relmae"].mean()) if not multi_df.empty else None,
        "multimodal_helps": (float(multi_df["relmae"].mean()) < float(price_df["relmae"].mean()))
            if not price_df.empty and not multi_df.empty else None,
    }


def generate_recommendations(horizon_analysis: dict, gaps: list[dict],
                             multimodal: dict) -> list[str]:
    """Hasilkan rekomendasi untuk arsitektur ARIF-Net."""
    recs = []

    # 1. Analisis horizon
    short_h = [h for h, a in horizon_analysis.items() if h <= 7]
    long_h = [h for h, a in horizon_analysis.items() if h >= 90]

    short_avg = np.mean([horizon_analysis[h]["avg_relmae"] for h in short_h]) if short_h else 0
    long_avg = np.mean([horizon_analysis[h]["avg_relmae"] for h in long_h]) if long_h else 0

    if long_avg > short_avg * 1.2:
        recs.append("HORIZON PANJANG SULIT: Model gagal di horizon ≥90 hari. "
                     "Pertimbangkan arsitektur multi-scale (misal: branch terpisah "
                     "untuk horizon pendek vs panjang).")

    # 2. Analisis multimodal
    if multimodal.get("multimodal_helps"):
        recs.append("MULTIMODAL MEMBANTU: Fitur eksternal (iklim, berita, BBM) "
                     "membantu prediksi. ARIF-Net harus memanfaatkan semua modalitas "
                     "dengan mekanisme fusion yang efektif.")
    else:
        recs.append("MULTIMODAL MARGINAL: Fitur eksternal belum banyak membantu. "
                     "Pertimbangkan mekanisme attention/gating untuk seleksi fitur "
                     "yang lebih cerdas.")

    # 3. Analisis gap
    if gaps:
        recs.append(f"DITEMUKAN {len(gaps)} GAP: Ada {len(gaps)} skenario (komoditas×horizon) "
                     "di mana semua model gagal. ARIF-Net perlu menangani skenario ini.")

    # 4. Rekomendasi arsitektur umum
    recs.append("SARAN ARSITEKTUR: Berdasarkan analisis, pertimbangkan komponen: "
                "(1) Backbone temporal (LSTM/GRU/TCN) untuk harga, "
                "(2) Gated fusion untuk modalitas eksternal, "
                "(3) Horizon-aware head untuk multi-horizon, "
                "(4) Residual connection ke persistence baseline.")

    return recs


def generate_report_md(horizon: dict, commodity: dict, gaps: list,
                       multimodal: dict, recs: list, df: pd.DataFrame) -> str:
    """Generate laporan markdown."""
    lines = [
        "# 🔍 ARIF-Net -- Gap Analysis Report",
        "",
        "## 1. Ringkasan Performa Model",
        "",
    ]

    # Model ranking
    if not df.empty:
        model_avg = df.groupby("model")["relmae"].mean().sort_values()
        lines.append("| Rank | Model | Avg relMAE |")
        lines.append("|---:|---|---:|")
        for rank, (model, relmae) in enumerate(model_avg.items(), 1):
            marker = " ✓" if relmae < 1.0 else " ✗"
            lines.append(f"| {rank} | {model} | {relmae:.4f}{marker} |")
        lines.append("")

    # Horizon analysis
    lines.extend(["## 2. Analisis Per Horizon", "", "| Horizon | Avg relMAE | Best Model | relMAE | Models < 1.0 |",
                   "|---:|---:|---|---:|---:|"])
    for h, a in sorted(horizon.items()):
        lines.append(f"| {h}d | {a['avg_relmae']:.4f} | {a['best_model']} | {a['best_relmae']:.4f} | "
                     f"{a['n_models_beating_persistence']} |")
    lines.append("")

    # Gaps
    lines.extend(["## 3. Celah (Gap) yang Ditemukan", ""])
    if gaps:
        lines.append("| Komoditas | Horizon | Best relMAE | Issue |")
        lines.append("|---|---:|---:|---|")
        for g in gaps:
            lines.append(f"| {g['commodity_name']} | {g['h']}d | {g['best_relmae']:.4f} | {g['issue']} |")
    else:
        lines.append("> Tidak ada gap kritis ditemukan.")
    lines.append("")

    # Multimodal
    lines.extend(["## 4. Multimodal vs Harga Saja", ""])
    if multimodal.get("price_only_avg_relmae") is not None:
        lines.append(f"- Harga saja: relMAE = {multimodal['price_only_avg_relmae']:.4f}")
        lines.append(f"- Multimodal: relMAE = {multimodal['multimodal_avg_relmae']:.4f}")
        lines.append(f"- **{'Multimodal membantu ✓' if multimodal['multimodal_helps'] else 'Multimodal belum membantu ✗'}**")
    lines.append("")

    # Recommendations
    lines.extend(["## 5. Rekomendasi untuk ARIF-Net", ""])
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Loading results...")

    df = load_all_results()
    if df.empty:
        logging.error("Tidak ada hasil training! Jalankan training dulu.")
        return 1

    logging.info("Loaded %d result rows from %d models", len(df), df["model"].nunique())

    # Analisis
    horizon = analyze_by_horizon(df)
    commodity = analyze_by_commodity(df)
    gaps = find_gaps(df)
    multimodal = multimodal_vs_price_only(df)
    recs = generate_recommendations(horizon, gaps, multimodal)

    # Save
    GAP_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "step": "gap_analysis",
        "n_models": int(df["model"].nunique()),
        "n_results": len(df),
        "horizon_analysis": horizon,
        "commodity_analysis": commodity,
        "gaps": gaps,
        "multimodal_comparison": multimodal,
        "recommendations": recs,
    }
    (GAP_DIR / "gap_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = generate_report_md(horizon, commodity, gaps, multimodal, recs, df)
    (GAP_DIR / "gap_analysis_report.md").write_text(md, encoding="utf-8")

    logging.info("Gap analysis selesai -> %s", GAP_DIR)
    logging.info("Gaps: %d skenario", len(gaps))
    for rec in recs:
        logging.info("  -> %s", rec[:80])

    return 0


if __name__ == "__main__":
    sys.exit(main())
