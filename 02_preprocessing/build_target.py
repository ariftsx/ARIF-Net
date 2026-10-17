"""
ARIF-Net -- Build target r(t,h) = (P(t+h) − P(t)) / P(t).

Untuk setiap origin (observasi PIHPS), cari label = observasi pertama ≥ t+h,
gap ≤ 7 hari, ≤ data_end. Flag holiday_carry & spike_reversal (deskriptif).

Output: data/processed/target/target_dataset.csv

Contoh:
  python -m 02_preprocessing.build_target [--commodity com_3]
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .utils import (
    COMMODITIES, DATA_PROCESSED, HORIZONS,
    load_project_config, now_utc, save_json,
)

OUT = DATA_PROCESSED / "target"


def label_index(dates: np.ndarray, h: int, tol: int,
                end: np.datetime64) -> tuple[np.ndarray, np.ndarray]:
    """Untuk setiap origin: (valid?, index_label). Label = obs pertama ≥ t+h; gap ≤ tol."""
    tgt = dates + np.timedelta64(h, "D")
    idx = np.searchsorted(dates, tgt, side="left")
    ok = idx < len(dates)
    idx = np.where(ok, idx, 0)
    ok &= ((dates[idx] - tgt).astype("timedelta64[D]").astype(int) <= tol) & (dates[idx] <= end)
    return ok, idx


def holiday_carry(dates: np.ndarray, prices: np.ndarray,
                  holidays: np.ndarray) -> np.ndarray:
    flag = np.zeros(len(dates), bool)
    if len(dates) > 1:
        flag[1:] = np.isin(dates[1:], holidays) & (prices[1:] == prices[:-1])
    return flag


def spike_reversal(prices: np.ndarray, jump: float = 0.15,
                   band: float = 0.05) -> np.ndarray:
    flag = np.zeros(len(prices), bool)
    if len(prices) >= 3:
        prev, cur, nxt = prices[:-2], prices[1:-1], prices[2:]
        flag[1:-1] = (np.abs(cur / prev - 1) >= jump) & (np.abs(nxt / prev - 1) <= band)
    return flag


def build_target_for_commodity(prices_df: pd.DataFrame, comcat_id: str,
                               commodity_name: str, cfg: dict) -> pd.DataFrame:
    """Build target untuk satu komoditas."""
    dates = prices_df["date"].values.astype("datetime64[D]")
    prices = prices_df["price"].values.astype(float)
    end = np.datetime64(cfg["data_end"])
    tol = cfg["target"]["label_tolerance_days"]
    spike_cfg = cfg["target"]["spike_reversal"]

    # Holidays (weekend + hari libur) -- simplified
    cal_path = DATA_PROCESSED.parent / "raw" / "calendar" / "calendar.csv"
    holidays = np.array([], dtype="datetime64[D]")
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        hol_dates = cal[cal["is_holiday"] == 1]["date"].values
        holidays = np.array(hol_dates, dtype="datetime64[D]")

    hc = holiday_carry(dates, prices, holidays)
    sr = spike_reversal(prices, spike_cfg["jump"], spike_cfg["revert_band"])

    parts = []
    for h in HORIZONS:
        ok, li = label_index(dates, h, tol, end)
        o = np.flatnonzero(ok)
        l = li[ok]
        p0, p1 = prices[o], prices[l]
        parts.append(pd.DataFrame({
            "comcat_id": comcat_id,
            "commodity": commodity_name,
            "origin_date": dates[o],
            "h": h,
            "target_date_nominal": dates[o] + np.timedelta64(h, "D"),
            "label_date": dates[l],
            "label_gap_days": (dates[l] - (dates[o] + np.timedelta64(h, "D"))).astype(int),
            "price_origin": p0,
            "price_label": p1,
            "r": (p1 - p0) / p0,
            "origin_holiday_carry": hc[o],
            "origin_spike_reversal": sr[o],
        }))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commodity", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_project_config()
    comms = [args.commodity] if args.commodity else list(COMMODITIES)

    frames = []
    for cid in comms:
        prices_path = DATA_PROCESSED / "pihps" / f"prices_{cid}.csv"
        if not prices_path.exists():
            logging.warning("Prices not found for %s, skipping.", cid)
            continue
        df = pd.read_csv(prices_path, parse_dates=["date"])
        comm_name = next((c["name"] for c in cfg["commodities"] if c["comcat_id"] == cid), cid)
        target = build_target_for_commodity(df, cid, comm_name, cfg)
        frames.append(target)
        counts = target.groupby("h").size().to_dict()
        logging.info("%s: %d rows | per horizon: %s", cid, len(target), counts)

    if not frames:
        logging.error("No data to build target!")
        return 1

    result = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "target_dataset.csv", index=False)
    logging.info("Target: %d rows -> target_dataset.csv", len(result))

    save_json(OUT / "build_target_report.json", {
        "step": "build_target", "rows": len(result),
        "commodities": comms, "horizons": list(HORIZONS),
        "finished_at_utc": now_utc(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
