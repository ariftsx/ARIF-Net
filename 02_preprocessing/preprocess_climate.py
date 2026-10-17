"""
ARIF-Net -- Preprocessing data iklim.

Membaca raw JSON dari Open-Meteo -> normalisasi per lokasi per hari.
Agregasi Tier-1 mean untuk fitur model.

Output: data/processed/climate/climate_daily.csv
        data/processed/climate/climate_tier1_mean.csv

Contoh:
  python -m 02_preprocessing.preprocess_climate
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_PROCESSED, DATA_RAW, load_json, load_project_config, now_utc, save_json

RAW = DATA_RAW / "climate"
OUT = DATA_PROCESSED / "climate"

WEATHER_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_max",
    "et0_fao_evapotranspiration", "sunshine_duration",
]


def parse_openmeteo_json(path: Path) -> pd.DataFrame:
    """Parse satu file Open-Meteo JSON -> DataFrame."""
    data = json.loads(path.read_text(encoding="utf-8"))
    daily = data.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return pd.DataFrame()
    records = {"date": times}
    for var in WEATHER_VARS:
        records[var] = daily.get(var, [None] * len(times))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_project_config()
    locs_cfg = load_json(Path(__file__).resolve().parents[1] / "configs" / "locations.json")
    locations = locs_cfg["locations"]

    all_frames = []
    for loc in locations:
        lid = loc["location_id"]
        files = sorted(RAW.glob(f"{lid}_*.json"))
        if not files:
            logging.warning("No data for %s", lid)
            continue
        for f in files:
            df = parse_openmeteo_json(f)
            if df.empty:
                continue
            df.insert(0, "location_id", lid)
            df.insert(1, "tier", loc["tier"])
            all_frames.append(df)
        logging.info("%-15s: %d file", lid, len(files))

    if not all_frames:
        logging.error("Tidak ada data iklim!")
        return 1

    combined = pd.concat(all_frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "climate_daily.csv", index=False)
    logging.info("Climate daily: %d baris -> climate_daily.csv", len(combined))

    # Tier 1 mean
    tier1 = combined[combined["tier"] == 1].copy()
    if not tier1.empty:
        tier1_mean = tier1.groupby("date")[WEATHER_VARS].mean().reset_index()
        tier1_mean.to_csv(OUT / "climate_tier1_mean.csv", index=False)
        logging.info("Tier-1 mean: %d hari -> climate_tier1_mean.csv", len(tier1_mean))

    # NaN report
    nan_report = {}
    for var in WEATHER_VARS:
        nan_count = int(combined[var].isna().sum())
        nan_report[var] = {"nan_count": nan_count, "nan_pct": round(nan_count / len(combined) * 100, 2)}

    save_json(OUT / "preprocess_climate_report.json", {
        "step": "preprocess_climate",
        "n_locations": len(set(combined["location_id"])),
        "n_rows": len(combined),
        "nan_report": nan_report,
        "finished_at_utc": now_utc(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
