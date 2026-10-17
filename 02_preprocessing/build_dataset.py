"""
ARIF-Net -- Build fitur multimodal & gabungkan ke dataset final siap model.

Menggabungkan semua preprocessed data:
  1. Target r(t,h) dari build_target
  2. Fitur harga (lag, MA, volatilitas, momentum)
  3. Fitur iklim (suhu, hujan, kelembapan -- tier1 mean)
  4. Fitur berita (sentimen, count, topik)
  5. Fitur BBM/logistik
  6. Fitur kalender (hari libur, Ramadan, dll)

Output:
  data/final/dataset_{fold}.csv           -- per fold (train/val/test split)
  data/final/dataset_full.csv             -- gabungan semua
  data/final/feature_registry.csv         -- daftar fitur + metadata

Contoh:
  python -m 02_preprocessing.build_dataset [--fold F1]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    COMMODITIES, COMMODITY_NAMES, DATA_FINAL, DATA_PROCESSED, FOLDS, HORIZONS,
    load_project_config, now_utc, save_json,
)


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Bangun fitur teknikal dari riwayat harga per komoditas."""
    features = prices[["date", "price"]].copy()

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        features[f"price_lag_{lag}"] = features["price"].shift(lag)

    # Return features
    for lag in [1, 3, 7, 14, 30]:
        features[f"return_{lag}d"] = features["price"].pct_change(lag)

    # Moving averages
    for window in [5, 10, 20, 30, 60, 90]:
        features[f"price_ma_{window}"] = features["price"].rolling(window, min_periods=1).mean()
        features[f"price_std_{window}"] = features["price"].rolling(window, min_periods=1).std()

    # Ratio to MA
    for window in [5, 20, 60]:
        ma_col = f"price_ma_{window}"
        features[f"price_ratio_ma_{window}"] = features["price"] / features[ma_col].replace(0, np.nan)

    # Volatilitas (std of returns)
    for window in [5, 20]:
        features[f"volatility_{window}d"] = features["return_1d"].rolling(window, min_periods=1).std()

    # Momentum
    features["momentum_7d"] = features["price"] - features["price"].shift(7)
    features["momentum_30d"] = features["price"] - features["price"].shift(30)

    return features


def merge_features(target: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Gabungkan semua fitur ke target dataset."""
    df = target.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])

    # ── 1. Price features ──
    for cid in COMMODITIES:
        prices_path = DATA_PROCESSED / "pihps" / f"prices_{cid}.csv"
        if not prices_path.exists():
            continue
        prices = pd.read_csv(prices_path, parse_dates=["date"])
        pfeat = build_price_features(prices)
        pfeat_cols = [c for c in pfeat.columns if c not in ("date", "price")]
        pfeat = pfeat.rename(columns={"date": "origin_date"})
        mask = df["comcat_id"] == cid
        df.loc[mask] = df.loc[mask].merge(pfeat[["origin_date"] + pfeat_cols],
                                           on="origin_date", how="left", suffixes=("", "_dup"))
        # Drop duplicate columns
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        df.drop(columns=dup_cols, inplace=True, errors="ignore")

    # ── 2. Climate features ──
    climate_path = DATA_PROCESSED / "climate" / "climate_tier1_mean.csv"
    if climate_path.exists():
        climate = pd.read_csv(climate_path, parse_dates=["date"])
        climate = climate.rename(columns={"date": "origin_date"})
        clim_cols = [c for c in climate.columns if c != "origin_date"]
        clim_renamed = {c: f"clim_{c}" for c in clim_cols}
        climate = climate.rename(columns=clim_renamed)
        df = df.merge(climate, on="origin_date", how="left")

    # ── 3. News features ──
    news_path = DATA_PROCESSED / "news" / "news_features.csv"
    if news_path.exists():
        news = pd.read_csv(news_path, parse_dates=["date"])
        news = news.rename(columns={"date": "origin_date"})
        news_cols = [c for c in news.columns if c != "origin_date"]
        df = df.merge(news, on="origin_date", how="left")
        # Fill missing news days with 0
        for c in news_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

    # ── 4. BBM features ──
    bbm_path = DATA_PROCESSED.parent / "raw" / "bbm" / "bbm_prices.csv"
    if bbm_path.exists():
        bbm = pd.read_csv(bbm_path, parse_dates=["date"])
        bbm = bbm.rename(columns={"date": "origin_date"})
        bbm_cols = [c for c in bbm.columns if c not in ("origin_date", "effective_date")]
        bbm_renamed = {c: f"bbm_{c}" for c in bbm_cols}
        bbm = bbm[["origin_date"] + bbm_cols].rename(columns=bbm_renamed)
        df = df.merge(bbm, on="origin_date", how="left")
        df[list(bbm_renamed.values())] = df[list(bbm_renamed.values())].ffill()

    # ── 5. Calendar features ──
    cal_path = DATA_PROCESSED.parent / "raw" / "calendar" / "calendar.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path, parse_dates=["date"])
        cal = cal.rename(columns={"date": "origin_date"})
        cal_cols = [c for c in cal.columns if c not in ("origin_date", "day_name", "year")]
        cal_renamed = {c: f"cal_{c}" for c in cal_cols}
        cal = cal[["origin_date"] + cal_cols].rename(columns=cal_renamed)
        df = df.merge(cal, on="origin_date", how="left")

    # ── 6. Horizon features ──
    df["h_log"] = np.log(df["h"])
    for hi, hval in enumerate(HORIZONS):
        df[f"h_is_{hval}"] = (df["h"] == hval).astype(int)

    return df


def split_data(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Split kronologis berdasarkan fold boundaries."""
    boundaries = cfg["validation"]["fold_boundaries"]
    splits = {}
    for fold in FOLDS:
        boundary = pd.Timestamp(boundaries[fold])
        train = df[df["origin_date"] < boundary].copy()
        val = df[(df["origin_date"] >= boundary)].copy()
        # For folds before test, val ends at next boundary
        fold_idx = list(FOLDS).index(fold)
        if fold_idx < len(FOLDS) - 1:
            next_boundary = pd.Timestamp(boundaries[FOLDS[fold_idx + 1]])
            val = val[val["origin_date"] < next_boundary]
        splits[fold] = {"train": train, "val": val}
    # Test split
    test_start = pd.Timestamp(cfg["validation"]["test_start"])
    splits["test"] = df[df["origin_date"] >= test_start].copy()
    return splits


def build_feature_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Buat registry fitur dengan metadata."""
    meta_cols = {"comcat_id", "commodity", "origin_date", "h",
                 "target_date_nominal", "label_date", "label_gap_days",
                 "price_origin", "price_label", "r",
                 "origin_holiday_carry", "origin_spike_reversal"}
    records = []
    for col in df.columns:
        if col in meta_cols:
            group = "meta"
        elif col.startswith("price_") or col.startswith("return_") or col.startswith("volatility_") or col.startswith("momentum_"):
            group = "PRICE"
        elif col.startswith("clim_"):
            group = "CLIMATE"
        elif col.startswith("news_") or col.startswith("sentiment_") or col.startswith("body_"):
            group = "NEWS"
        elif col.startswith("bbm_"):
            group = "BBM"
        elif col.startswith("cal_"):
            group = "CALENDAR"
        elif col.startswith("h_"):
            group = "HORIZON"
        else:
            group = "other"
        records.append({
            "feature": col, "group": group, "dtype": str(df[col].dtype),
            "n_nan": int(df[col].isna().sum()),
            "nan_pct": round(df[col].isna().mean() * 100, 2),
        })
    return pd.DataFrame(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=None, choices=list(FOLDS))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_project_config()

    # Load target
    target_path = DATA_PROCESSED / "target" / "target_dataset.csv"
    if not target_path.exists():
        logging.error("Target dataset not found! Run build_target first.")
        return 1
    target = pd.read_csv(target_path, parse_dates=["origin_date"])
    logging.info("Target loaded: %d rows", len(target))

    # Merge features
    logging.info("Merging multimodal features...")
    dataset = merge_features(target, cfg)
    logging.info("Dataset: %d rows × %d columns", len(dataset), len(dataset.columns))

    # Feature registry
    registry = build_feature_registry(dataset)
    DATA_FINAL.mkdir(parents=True, exist_ok=True)
    registry.to_csv(DATA_FINAL / "feature_registry.csv", index=False)

    # Feature groups summary
    group_counts = registry.groupby("group").size().to_dict()
    logging.info("Feature groups: %s", group_counts)

    # Save full dataset
    dataset.to_csv(DATA_FINAL / "dataset_full.csv", index=False)
    logging.info("Full dataset -> data/final/dataset_full.csv")

    # Split and save
    splits = split_data(dataset, cfg)
    for fold, data in splits.items():
        if isinstance(data, dict):
            for part, df_part in data.items():
                path = DATA_FINAL / f"dataset_{fold}_{part}.csv"
                df_part.to_csv(path, index=False)
                logging.info("%s_%s: %d rows -> %s", fold, part, len(df_part), path.name)
        else:
            path = DATA_FINAL / f"dataset_{fold}.csv"
            data.to_csv(path, index=False)
            logging.info("%s: %d rows -> %s", fold, len(data), path.name)

    save_json(DATA_FINAL / "build_dataset_report.json", {
        "step": "build_dataset",
        "total_rows": len(dataset),
        "total_features": len(dataset.columns),
        "feature_groups": group_counts,
        "commodities": list(COMMODITIES),
        "horizons": list(HORIZONS),
        "finished_at_utc": now_utc(),
    })
    logging.info("BUILD DATASET SELESAI ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
