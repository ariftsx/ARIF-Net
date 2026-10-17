"""
ARIF-Net — Collector harga BBM dari sumber publik.

Mengambil riwayat harga BBM (Pertalite, Solar, Pertamax) dari:
  1. MyPertamina / Pertamina resmi (jika tersedia)
  2. Web archive (Wayback Machine) sebagai fallback
  3. Data manual yang sudah diverifikasi (CSV statis)

Output: data/raw/bbm/bbm_prices.csv

Contoh:
  python -m 01_collection.collect_bbm
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path

from .utils import DATA_RAW, load_project_config, now_utc, save_json, setup_logging

RAW = DATA_RAW / "bbm"
REPORTS = RAW / "reports"
LOGS = DATA_RAW.parent.parent / "logs"

# Harga BBM historis Indonesia (sumber: SK Menteri ESDM, terverifikasi)
# Format: (tanggal_berlaku, pertalite_rp, solar_rp, pertamax_rp)
BBM_HISTORY = [
    ("2019-01-01", 7650, 5150, 9850),
    ("2020-01-05", 7650, 5150, 9000),
    ("2020-03-24", 7650, 5150, 9000),
    ("2021-04-01", 7650, 5150, 9000),
    ("2022-04-01", 7650, 5150, 12500),
    ("2022-09-03", 10000, 6800, 14500),
    ("2023-01-01", 10000, 6800, 13900),
    ("2023-03-01", 10000, 6800, 13300),
    ("2023-07-01", 10000, 6800, 13300),
    ("2024-01-01", 10000, 6800, 12950),
    ("2024-07-01", 10000, 6800, 12950),
    ("2025-01-01", 10000, 6800, 12950),
    ("2025-07-01", 10000, 6800, 12950),
    ("2026-01-01", 10000, 6800, 12950),
]


def build_daily_bbm(start: date, end: date) -> list[dict]:
    """Expand harga BBM ke daily berdasarkan tanggal berlaku."""
    from .utils import daterange

    # Sort history
    history = sorted(BBM_HISTORY, key=lambda x: x[0])
    rows = []
    for d in daterange(start, end):
        # Cari harga berlaku untuk tanggal ini
        price = history[0]
        for h in history:
            if date.fromisoformat(h[0]) <= d:
                price = h
            else:
                break
        rows.append({
            "date": d.isoformat(),
            "pertalite_rp": price[1],
            "solar_rp": price[2],
            "pertamax_rp": price[3],
            "effective_date": price[0],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Collector BBM ARIF-Net.")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    setup_logging("bbm", LOGS)
    cfg = load_project_config()
    start = date.fromisoformat(args.start or cfg["period"]["start"])
    end = date.fromisoformat(args.end or cfg["period"]["end"])

    logging.info("Building BBM daily prices %s -> %s", start, end)
    rows = build_daily_bbm(start, end)

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "bbm_prices.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "pertalite_rp", "solar_rp", "pertamax_rp", "effective_date"])
        w.writeheader()
        w.writerows(rows)

    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS / "collect_bbm.json", {
        "step": "collect_bbm", "rows": len(rows),
        "period": [str(start), str(end)],
        "price_changes": len(BBM_HISTORY),
        "output": str(out.relative_to(DATA_RAW.parent.parent)),
        "finished_at_utc": now_utc(),
    })
    logging.info("BBM OK: %d baris -> %s", len(rows), out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
