"""
ARIF-Net -- Preprocessing harga komoditas PIHPS.

Membaca raw JSON chunks dari collector PIHPS -> normalisasi ke DataFrame harian
per komoditas, satu baris = satu tanggal, satu kolom = harga (Rp).

Output: data/processed/pihps/prices_{comcat_id}.csv
        data/processed/pihps/prices_combined.csv

Contoh:
  python -m 02_preprocessing.preprocess_pihps
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .utils import (
    COMMODITIES, COMMODITY_NAMES, DATA_PROCESSED, DATA_RAW,
    load_project_config, now_utc, save_json,
)

RAW = DATA_RAW / "pihps"
OUT = DATA_PROCESSED / "pihps"
LOGS = DATA_RAW.parent.parent / "logs"


def parse_pihps_json(path: Path, market_name: str = "Kramatjati",
                     market_level: int = 3) -> list[tuple[date, float | None]]:
    """Parse satu file raw JSON PIHPS -> list (date, price_or_None)."""
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    data = payload.get("data", [])
    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        lvl = row.get("level", 0)
        if market_name.lower() in name.lower() and int(lvl) == market_level:
            items = []
            for k, v in row.items():
                try:
                    d = date.fromisoformat(k)
                except (ValueError, TypeError):
                    continue
                val = str(v).strip().replace(",", "").replace(".", "")
                if val in ("-", "", "0"):
                    items.append((d, None))
                else:
                    try:
                        items.append((d, float(val)))
                    except ValueError:
                        items.append((d, None))
            return sorted(items, key=lambda x: x[0])
    return []


def build_prices(comcat_id: str) -> pd.DataFrame:
    """Gabungkan semua raw chunks untuk satu komoditas -> DataFrame (date, price)."""
    records = []
    for f in sorted(RAW.glob(f"{comcat_id}_*.json")):
        if "smoke" in f.name:
            continue
        items = parse_pihps_json(f)
        for d, p in items:
            if p is not None and d.weekday() < 5:  # Hanya hari kerja
                records.append({"date": d.isoformat(), "price": p})
    if not records:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_project_config()

    OUT.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for comm in cfg["commodities"]:
        cid = comm["comcat_id"]
        logging.info("Processing PIHPS %s (%s)...", cid, comm["name"])
        df = build_prices(cid)
        if df.empty:
            logging.warning("%s: tidak ada data!", cid)
            continue
        df.insert(0, "comcat_id", cid)
        df.insert(1, "commodity", comm["name"])

        out_path = OUT / f"prices_{cid}.csv"
        df.to_csv(out_path, index=False)
        logging.info("%s: %d baris, %s -> %s, harga %.0f-%.0f",
                     cid, len(df), df["date"].min().date(), df["date"].max().date(),
                     df["price"].min(), df["price"].max())
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUT / "prices_combined.csv", index=False)
        logging.info("Combined: %d baris -> prices_combined.csv", len(combined))

    save_json(OUT / "preprocess_pihps_report.json", {
        "step": "preprocess_pihps",
        "commodities": {c["comcat_id"]: len(build_prices(c["comcat_id"])) for c in cfg["commodities"]},
        "finished_at_utc": now_utc(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
