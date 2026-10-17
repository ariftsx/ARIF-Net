"""
ARIF-Net -- Collector data iklim dari Open-Meteo API.

Mengambil data cuaca harian untuk 18 kabupaten pemasok:
  - temperature_2m_mean/max/min
  - precipitation_sum
  - relative_humidity_2m_mean
  - wind_speed_10m_max
  - et0_fao_evapotranspiration (evapotranspirasi)
  - sunshine_duration

Periode: 2019-01-01 -> 2026-09-30
API: Open-Meteo Historical Weather (gratis, tanpa API key)

Contoh:
  python -m 01_collection.collect_climate [--location brebes] [--start 2019-01-01]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import date
from pathlib import Path

import requests

from .utils import (
    DATA_RAW, HTTPClient, append_csv, load_json, load_locations,
    load_project_config, now_utc, save_json, setup_logging, sha256_bytes, stamp,
)

RAW = DATA_RAW / "climate"
MANIFEST = RAW / "manifest.csv"
REPORTS = RAW / "reports"
LOGS = DATA_RAW.parent.parent / "logs"

MANIFEST_FIELDS = [
    "location_id", "kabupaten", "lat", "lon", "start_date", "end_date",
    "http_status", "n_days", "raw_file", "raw_sha256", "check", "error",
    "collected_at_utc",
]

WEATHER_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_max",
    "et0_fao_evapotranspiration", "sunshine_duration",
]

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_location(http: HTTPClient, loc: dict, start: date, end: date,
                   log: logging.Logger) -> dict:
    """Fetch data cuaca untuk satu lokasi dari Open-Meteo."""
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(WEATHER_VARS),
        "timezone": "Asia/Jakarta",
    }
    r = http.get(OPENMETEO_URL, params=params, label=loc["location_id"])
    row = {
        "location_id": loc["location_id"],
        "kabupaten": loc["kabupaten"],
        "lat": loc["lat"], "lon": loc["lon"],
        "start_date": str(start), "end_date": str(end),
        "collected_at_utc": now_utc(),
    }
    if r is None:
        row.update({"http_status": "EXC", "check": "FAIL", "error": "retries habis"})
        return row

    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{loc['location_id']}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    f.write_bytes(r.content)
    row["raw_file"] = str(f.relative_to(DATA_RAW.parent.parent))
    row["raw_sha256"] = sha256_bytes(r.content)
    row["http_status"] = r.status_code

    if r.status_code != 200:
        row.update({"check": "FAIL", "error": f"HTTP {r.status_code}"})
        return row

    try:
        data = r.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        row["n_days"] = len(times)
        # Cek completeness
        missing = [v for v in WEATHER_VARS if v not in daily]
        if missing:
            row.update({"check": "WARN", "error": f"variabel hilang: {missing}"})
        else:
            row["check"] = "OK"
    except Exception as exc:
        row.update({"check": "FAIL", "error": str(exc)[:300]})

    return row


def done_locations() -> set[str]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open(encoding="utf-8") as fh:
        return {r["location_id"] for r in csv.DictReader(fh) if r["check"] in ("OK", "WARN")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Collector iklim ARIF-Net (Open-Meteo).")
    ap.add_argument("--location", default=None, help="location_id tunggal (default: semua)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    setup_logging("climate", LOGS)
    cfg = load_project_config()
    locs = load_locations()
    start = date.fromisoformat(args.start or cfg["period"]["start"])
    end = date.fromisoformat(args.end or cfg["period"]["end"])

    locations = locs["locations"]
    if args.location:
        locations = [l for l in locations if l["location_id"] == args.location]

    done = set() if args.redo else done_locations()
    todo = [l for l in locations if l["location_id"] not in done]
    logging.info("Lokasi: %d total, %d sudah, %d sisa", len(locations), len(done), len(todo))

    http = HTTPClient(delay=0.5, retries=3, timeout=60)
    n_ok = n_err = 0
    for loc in todo:
        row = fetch_location(http, loc, start, end, logging.getLogger(loc["location_id"]))
        append_csv(MANIFEST, MANIFEST_FIELDS, row)
        if row["check"] in ("OK", "WARN"):
            n_ok += 1
            logging.info("%-15s OK | %s hari", loc["location_id"], row.get("n_days", "?"))
        else:
            n_err += 1
            logging.error("%-15s FAIL | %s", loc["location_id"], row.get("error", ""))
        time.sleep(0.3)  # Be nice to free API

    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS / "collect_climate.json", {
        "step": "collect_climate", "ok": n_ok, "error": n_err,
        "period": [str(start), str(end)],
        "locations": [l["location_id"] for l in todo],
        "finished_at_utc": now_utc(),
    })
    logging.info("SELESAI: OK=%d ERROR=%d", n_ok, n_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
