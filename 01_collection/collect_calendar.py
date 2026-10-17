"""
ARIF-Net -- Generator kalender hari libur & hari besar Indonesia.

Membangun kalender harian 2019-2026 dengan fitur:
  - is_holiday (hari libur nasional)
  - is_weekend
  - is_ramadan (bulan puasa)
  - is_idul_fitri (Lebaran ±3 hari)
  - is_idul_adha
  - is_natal_tahun_baru (25 Des - 1 Jan)
  - day_of_week, month, quarter

Sumber: Keputusan Presiden tentang Hari Libur Nasional (manual, terverifikasi).

Contoh:
  python -m 01_collection.collect_calendar
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from .utils import DATA_RAW, daterange, load_project_config, now_utc, save_json, setup_logging

RAW = DATA_RAW / "calendar"
LOGS = DATA_RAW.parent.parent / "logs"

# Tanggal libur nasional utama (Islam mengikuti penanggalan Hijriah, tanggal bervariasi)
# Format: list of (date_str, nama, jenis)
HOLIDAYS = {
    # Idul Fitri (1 Syawal, ±cuti bersama)
    "idul_fitri": [
        ("2019-06-05", "2019-06-06"), ("2020-05-24", "2020-05-25"),
        ("2021-05-13", "2021-05-14"), ("2022-05-02", "2022-05-03"),
        ("2023-04-22", "2023-04-23"), ("2024-04-10", "2024-04-11"),
        ("2025-03-31", "2025-04-01"), ("2026-03-20", "2026-03-21"),
    ],
    # Idul Adha (10 Dzulhijjah)
    "idul_adha": [
        "2019-08-11", "2020-07-31", "2021-07-20", "2022-07-10",
        "2023-06-29", "2024-06-17", "2025-06-07", "2026-05-27",
    ],
    # Ramadan start (1 Ramadan, ±30 hari)
    "ramadan_start": [
        "2019-05-06", "2020-04-24", "2021-04-13", "2022-04-02",
        "2023-03-23", "2024-03-12", "2025-03-01", "2026-02-18",
    ],
    # Natal & Tahun Baru (tetap)
    "natal": "12-25",
    "tahun_baru": "01-01",
    # Hari kemerdekaan
    "kemerdekaan": "08-17",
}


def build_calendar(start: date, end: date) -> list[dict]:
    """Build kalender harian dengan fitur temporal."""
    # Parse holiday dates
    idul_fitri_dates = set()
    for pair in HOLIDAYS["idul_fitri"]:
        d1, d2 = date.fromisoformat(pair[0]), date.fromisoformat(pair[1])
        for delta in range(-3, 4):
            idul_fitri_dates.add(d1 + timedelta(days=delta))
            idul_fitri_dates.add(d2 + timedelta(days=delta))

    idul_adha_dates = set()
    for ds in HOLIDAYS["idul_adha"]:
        d = date.fromisoformat(ds)
        for delta in range(-1, 2):
            idul_adha_dates.add(d + timedelta(days=delta))

    ramadan_dates = set()
    for ds in HOLIDAYS["ramadan_start"]:
        d = date.fromisoformat(ds)
        for delta in range(30):
            ramadan_dates.add(d + timedelta(days=delta))

    fixed_holidays = set()
    for y in range(start.year, end.year + 1):
        fixed_holidays.add(date(y, 12, 25))  # Natal
        fixed_holidays.add(date(y, 1, 1))    # Tahun Baru
        fixed_holidays.add(date(y, 8, 17))   # Kemerdekaan

    all_holidays = idul_fitri_dates | idul_adha_dates | fixed_holidays

    rows = []
    for d in daterange(start, end):
        rows.append({
            "date": d.isoformat(),
            "day_of_week": d.weekday(),
            "day_name": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][d.weekday()],
            "month": d.month,
            "quarter": (d.month - 1) // 3 + 1,
            "year": d.year,
            "is_weekend": int(d.weekday() >= 5),
            "is_holiday": int(d in all_holidays),
            "is_ramadan": int(d in ramadan_dates),
            "is_idul_fitri": int(d in idul_fitri_dates),
            "is_idul_adha": int(d in idul_adha_dates),
            "is_natal_tahun_baru": int(
                (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2)
            ),
            "is_kemerdekaan": int(d.month == 8 and d.day == 17),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Generator kalender ARIF-Net.")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    setup_logging("calendar", LOGS)
    cfg = load_project_config()
    start = date.fromisoformat(args.start or cfg["period"]["start"])
    end = date.fromisoformat(args.end or cfg["period"]["end"])

    logging.info("Building calendar %s -> %s", start, end)
    rows = build_calendar(start, end)

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "calendar.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_holidays = sum(1 for r in rows if r["is_holiday"])
    n_ramadan = sum(1 for r in rows if r["is_ramadan"])
    logging.info("Calendar OK: %d hari, %d libur, %d Ramadan -> %s",
                 len(rows), n_holidays, n_ramadan, out.name)

    save_json(RAW / "reports" / "collect_calendar.json", {
        "step": "collect_calendar", "rows": len(rows),
        "holidays": n_holidays, "ramadan_days": n_ramadan,
        "output": str(out.relative_to(DATA_RAW.parent.parent)),
        "finished_at_utc": now_utc(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
