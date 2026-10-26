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
    """Fitur harga BEBAS SKALA dari riwayat harga per komoditas (hanya data <= t; jendela = observasi terlapor).

    Level harga mentah (lag/MA dalam Rp) sengaja tidak dipakai: tidak stasioner, sehingga model linear/neural
    mengekstrapolasi saat harga pindah ke level baru (mis. beras 2023-2024). Satu fitur level tetap ada:
    price_level = log harga (PRICE_FEATURE_CONTRACT lama: ret_k, statistik bergulir relatif, price_level).
    """
    p = prices["price"].astype(float)
    features = prices[["date", "price"]].copy()
    features["price_level"] = np.log(p)

    # Return k observasi (setara lag harga, tapi relatif)
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        features[f"return_{lag}d"] = p.pct_change(lag, fill_method=None)

    # Posisi terhadap rata-rata bergulir & dispersi relatif (CV)
    for window in [5, 10, 20, 30, 60, 90]:
        ma = p.rolling(window, min_periods=1).mean()
        features[f"price_ratio_ma_{window}"] = p / ma - 1
        features[f"price_cv_{window}"] = p.rolling(window, min_periods=2).std() / ma

    # Volatilitas (std return harian)
    for window in [5, 20]:
        features[f"volatility_{window}d"] = features["return_1d"].rolling(window, min_periods=2).std()

    # Level mentah (Rp): informatif untuk model pohon (tidak mengekstrapolasi); untuk model linear/neural
    # distandardisasi lalu dipotong ±5σ di 03_training/data.py.
    for lag in [1, 7, 30]:
        features[f"price_lag_{lag}"] = p.shift(lag)
    for window in [5, 20, 60]:
        features[f"price_ma_{window}"] = p.rolling(window, min_periods=1).mean()

    return features


DRIFT_WINDOWS = (365, 730)
DRIFT_SEASON_YEARS = 3
DRIFT_MIN_OBS = 20


def build_drift_features(prices: pd.DataFrame, origins: np.ndarray, horizons: np.ndarray) -> pd.DataFrame:
    """Fitur drift ARIF-Net (docs/ARIFNET_SPEC.md §2.4, G-3), HANYA dari harga <= t.

    drift_slope_W  : kemiringan Theil–Sen log-harga per hari atas observasi di (t − W, t]  (dikali h di model)
    drift_season   : rata-rata k=1..K [log P(t − 365k + h) − log P(t − 365k)], P = observasi terakhir <= tanggal itu;
                     karena h <= 365, kedua tanggal <= t (tanpa kebocoran)
    """
    from scipy.stats import theilslopes
    d = prices["date"].to_numpy("datetime64[D]")
    lp = np.log(prices["price"].to_numpy(float))
    day = d.astype("int64")

    slopes = {}
    for t in np.unique(origins):
        i = int(np.searchsorted(d, t, side="right"))
        row = {}
        for W in DRIFT_WINDOWS:
            j = int(np.searchsorted(d, t - np.timedelta64(W, "D"), side="right"))
            row[f"drift_slope_{W}"] = (theilslopes(lp[j:i], day[j:i])[0] if i - j >= DRIFT_MIN_OBS else np.nan)
        slopes[t] = row

    def asof(dates: np.ndarray) -> np.ndarray:
        k = np.searchsorted(d, dates, side="right") - 1
        return np.where(k >= 0, lp[np.maximum(k, 0)], np.nan)

    seas = np.full(len(origins), np.nan)
    acc, n = np.zeros(len(origins)), np.zeros(len(origins))
    for k in range(1, DRIFT_SEASON_YEARS + 1):
        base = origins - np.timedelta64(365 * k, "D")
        v = asof(base + horizons.astype("timedelta64[D]")) - asof(base)
        ok = np.isfinite(v) & (base >= d[0])
        acc[ok] += v[ok]
        n[ok] += 1
    seas[n > 0] = acc[n > 0] / n[n > 0]

    out = pd.DataFrame([slopes[t] for t in origins])
    out.insert(0, "origin_date", origins)
    out.insert(1, "h", horizons)
    out["drift_season"] = seas
    return out


def merge_features(target: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Gabungkan semua fitur ke target dataset."""
    df = target.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])

    # ── 1. Price features ──
    # Merge per komoditas lalu concat. Jangan assign balik via df.loc[mask]:
    # merge() mereset index (baris tidak sejajar -> NaN) dan kolom baru dibuang.
    parts = []
    for cid, part in df.groupby("comcat_id", sort=False):
        prices_path = DATA_PROCESSED / "pihps" / f"prices_{cid}.csv"
        if cid in COMMODITIES and prices_path.exists():
            prices = pd.read_csv(prices_path, parse_dates=["date"])
            pfeat = build_price_features(prices)
            pfeat_cols = [c for c in pfeat.columns if c not in ("date", "price")]
            pfeat = pfeat.rename(columns={"date": "origin_date"})
            part = part.merge(pfeat[["origin_date"] + pfeat_cols],
                              on="origin_date", how="left", validate="many_to_one")
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    # ── 2. Climate features ──
    climate_path = DATA_PROCESSED / "climate" / "climate_tier1_mean.csv"
    if climate_path.exists():
        climate = pd.read_csv(climate_path, parse_dates=["date"])
        # Data Contract §6 (P1-DG-17): iklim tersedia dengan jeda rilis, origin t memakai data <= t - lag.
        lag = int(cfg["availability"]["climate_lag_days"])
        climate["date"] = climate["date"] + pd.Timedelta(days=lag)
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
        # Harga BBM diatur pemerintah (fungsi tangga): level mentah hampir konstan di data latih lalu melompat,
        # sehingga dipakai bentuk relatif: perubahan log 30/365 hari dan lama sejak penyesuaian terakhir.
        bbm = pd.read_csv(bbm_path, parse_dates=["date"]).sort_values("date").set_index("date")
        bbm = bbm.asfreq("D").ffill()
        fuels = [c for c in bbm.columns if c != "effective_date"]
        feats = pd.DataFrame(index=bbm.index)
        for f in fuels:
            name = f.removesuffix("_rp")
            lv = np.log(bbm[f].astype(float))
            feats[f"bbm_{name}_chg_30d"] = lv - lv.shift(30)
            feats[f"bbm_{name}_chg_365d"] = lv - lv.shift(365)
            changed = lv.diff().fillna(0) != 0
            run_id = changed.cumsum()
            feats[f"bbm_{name}_days_since_change"] = (run_id.groupby(run_id).cumcount() / 365.0).to_numpy()
        feats = feats.reset_index().rename(columns={"date": "origin_date"})
        df = df.merge(feats, on="origin_date", how="left")

    # ── 5. Calendar features ──
    cal_path = DATA_PROCESSED.parent / "raw" / "calendar" / "calendar.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path, parse_dates=["date"])
        cal = cal.rename(columns={"date": "origin_date"})
        cal_cols = [c for c in cal.columns if c not in ("origin_date", "day_name", "year")]
        cal_renamed = {c: f"cal_{c}" for c in cal_cols}
        cal = cal[["origin_date"] + cal_cols].rename(columns=cal_renamed)
        df = df.merge(cal, on="origin_date", how="left")
        # Kalender pada tanggal target t+h: diketahui sebelumnya (known future input, dipakai TFT).
        fut = cal.rename(columns={"origin_date": "target_date_nominal",
                                  **{v: f"tgt_{v}" for v in cal_renamed.values()}})
        df["target_date_nominal"] = pd.to_datetime(df["target_date_nominal"])
        df = df.merge(fut, on="target_date_nominal", how="left")

    # ── 6. Drift features (ARIF-Net G-3; tidak masuk set fitur baseline) ──
    parts = []
    for cid, part in df.groupby("comcat_id", sort=False):
        prices = pd.read_csv(DATA_PROCESSED / "pihps" / f"prices_{cid}.csv", parse_dates=["date"])
        drift = build_drift_features(prices, part["origin_date"].to_numpy("datetime64[D]"),
                                     part["h"].to_numpy(int))
        drift["origin_date"] = pd.to_datetime(drift["origin_date"])
        parts.append(part.merge(drift.drop_duplicates(["origin_date", "h"]), on=["origin_date", "h"],
                                how="left", validate="many_to_one"))
    df = pd.concat(parts, ignore_index=True)

    # ── 7. Horizon features ──
    df["h_log"] = np.log(df["h"])
    for hi, hval in enumerate(HORIZONS):
        df[f"h_is_{hval}"] = (df["h"] == hval).astype(int)

    return df


def split_data(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Split kronologis rolling-origin dengan purging label.

    Fold k: train = origin < b_k DAN label < b_k; val = b_k <= origin < b_{k+1} DAN label < test_start.
    Final: train = origin < test_start DAN label < test_start; test = origin >= test_start, label <= test_end.
    Purging memastikan tidak ada label train yang melihat periode val, dan label val tidak masuk periode test.
    """
    v = cfg["validation"]
    folds = list(v["folds"])
    bounds = [pd.Timestamp(v["fold_boundaries"][f]) for f in folds]
    test_start, test_end = pd.Timestamp(v["test_start"]), pd.Timestamp(v["test_end"])
    origin, label = df["origin_date"], pd.to_datetime(df["label_date"])
    splits = {}
    for i, fold in enumerate(folds):
        b, b_next = bounds[i], bounds[i + 1] if i + 1 < len(bounds) else test_start
        splits[fold] = {
            "train": df[(origin < b) & (label < b)].copy(),
            "val": df[(origin >= b) & (origin < b_next) & (label < test_start)].copy(),
        }
    splits["final"] = {"train": df[(origin < test_start) & (label < test_start)].copy()}
    splits["test"] = df[(origin >= test_start) & (label <= test_end)].copy()

    # Uji prospektif (docs/PROTOKOL_UJI_PROSPEKTIF.md): latih = origin & label <= train_end; uji = origin >= start,
    # label <= cutoff. Peran uji hanya terisi setelah data sesudah train_end dikumpulkan.
    p = cfg["prospective"]
    p_end, p_start, p_cut = (pd.Timestamp(p[k]) for k in ("train_end", "start", "cutoff"))
    splits["prospektif"] = {"train": df[(origin <= p_end) & (label <= p_end)].copy()}
    p_test = df[(origin >= p_start) & (label <= p_cut)].copy()
    if len(p_test):
        splits["prospektif"]["test"] = p_test
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
        elif col.startswith("tgt_cal_"):
            group = "CALENDAR_FUTURE"
        elif col.startswith("drift_"):
            group = "DRIFT"
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
        "splits": {f"{fold}_{part}" if isinstance(data, dict) else fold: len(d)
                   for fold, data in splits.items()
                   for part, d in (data.items() if isinstance(data, dict) else [("", data)])},
        "split_rule": "rolling-origin + purging label (lihat split_data)",
        "climate_lag_days": cfg["availability"]["climate_lag_days"],
        "commodities": list(COMMODITIES),
        "horizons": list(HORIZONS),
        "finished_at_utc": now_utc(),
    })
    logging.info("BUILD DATASET SELESAI ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
