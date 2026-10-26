"""
ARIF-Net -- Pemuat data SEMUA model training (satu pintu: data, split, target, dan preprocessing identik).

Peran data (configs/project.json `validation`):
  F1, F2   fold validasi (dataset_{F}_train.csv / dataset_{F}_val.csv), dipakai pencarian & validasi
  final    latih final (dataset_final_train.csv) -> TEST (dataset_test.csv); hanya dibuka oleh final_test.py

Aturan anti-kebocoran:
  * Imputasi (median) dan standardisasi (mean/std) fitur di-fit di data LATIH per komoditas saja.
  * σ_{c,h} (skala target) dan ambang shock Q95 dihitung dari data LATIH saja.
  * Sequence harga = L observasi terakhir <= t (hanya masa lalu); konteks boleh melintasi batas split ke masa lalu.
  * Sequence iklim (DIA-LSTM) memakai iklim tanggal d - lag (jeda rilis, configs/project.json `availability`).

Kelompok fitur:
  PRICE  price_*, return_*, volatility_*, momentum_*     CLIM  clim_*
  NEWS   news_*, sentiment_*, body_*                      BBM   bbm_*
  CAL    cal_*  (kalender pada origin t)                   CALF  tgt_cal_* (kalender pada tanggal target, known future)
  DRIFT  drift_* (ARIF-Net G-3; juga tersedia mentah sebagai x["DRIFT_RAW"], NaN → 0 = tanpa drift)

Peran prospektif (docs/PROTOKOL_UJI_PROSPEKTIF.md): latih = snapshot yang dikunci saat registrasi
(results/prospektif/train_snapshot.csv), uji = dataset_prospektif_test.csv; hanya dibuka oleh final_test.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Jumlah core fisik mesin yang menjalankan (bawaan PyTorch = core fisik), dibaca sebelum ada set_num_threads.
_PHYSICAL_CORES = torch.get_num_threads()


def resolve_threads(value=None) -> int:
    """`train.threads` = "auto" (default) → core fisik mesin ini; env ARIFNET_THREADS menimpa semuanya."""
    env = os.environ.get("ARIFNET_THREADS")
    if env:
        return max(1, int(env))
    if value in (None, "auto"):
        return max(1, _PHYSICAL_CORES)
    return max(1, int(value))

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA_FINAL = ROOT / "data" / "final"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

PROJECT = json.loads((CONFIGS / "project.json").read_text(encoding="utf-8"))
COMMODITIES = tuple(c["comcat_id"] for c in PROJECT["commodities"])
HORIZONS = tuple(PROJECT["horizons"])
VALIDATION_FOLDS = tuple(PROJECT["validation"]["folds"])
SEEDS = tuple(PROJECT["training"]["seeds"])
CLIMATE_LAG = int(PROJECT["availability"]["climate_lag_days"])

META = ["comcat_id", "commodity", "origin_date", "h", "target_date_nominal", "label_date", "label_gap_days",
        "price_origin", "price_label", "r", "origin_holiday_carry", "origin_spike_reversal"]
GROUPS = {
    "PRICE": ("price_", "return_", "volatility_", "momentum_"),
    "CLIM": ("clim_",),
    "NEWS": ("news_", "sentiment_", "body_"),
    "BBM": ("bbm_",),
    "CAL": ("cal_",),
    "CALF": ("tgt_cal_",),
    "DRIFT": ("drift_",),          # khusus ARIF-Net (G-3); tidak termasuk input baseline
}
MULTIMODAL = ("PRICE", "CLIM", "NEWS", "BBM", "CAL")      # input tabular "multimodal" (tanpa known future)
SEQ_PRICE_CHANNELS = 4                                     # rel_logp, ret_obs, gap_days/7, mask


def group_of(col: str) -> str | None:
    for g, prefixes in GROUPS.items():
        if col.startswith(prefixes) and not (g == "PRICE" and col in META):
            return g
    return None


def feature_columns(df: pd.DataFrame, groups=MULTIMODAL) -> list[str]:
    return [c for c in df.columns if c not in META and group_of(c) in groups]


# ───────────────────────────────────────────── baca split
def read_split(role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1/F2 -> (train, val). 'final' -> (final_train, test): HANYA untuk final_test.py sesudah kunci ditulis."""
    if role in VALIDATION_FOLDS:
        names = (f"dataset_{role}_train.csv", f"dataset_{role}_val.csv")
    elif role == "final":
        names = ("dataset_final_train.csv", "dataset_test.csv")
    elif role == "prospektif":
        names = (str(RESULTS / "prospektif" / "train_snapshot.csv"), "dataset_prospektif_test.csv")
    else:
        raise KeyError(f"peran data tidak dikenal: {role} (pilih {VALIDATION_FOLDS} atau final)")
    out = []
    for n in names:
        p = DATA_FINAL / n
        if not p.exists():
            raise FileNotFoundError(f"{p} tidak ada -- jalankan `python run.py preprocess`")
        out.append(pd.read_csv(p, parse_dates=["origin_date", "label_date", "target_date_nominal"]))
    tr, ev = out
    _check_purging(role, tr, ev)
    return tr, ev


def _check_purging(role: str, tr: pd.DataFrame, ev: pd.DataFrame) -> None:
    """Gagal keras bila label latih melewati awal periode evaluasi (kebocoran)."""
    start = ev["origin_date"].min()
    if (tr["label_date"] >= start).any() or (tr["origin_date"] >= start).any():
        raise RuntimeError(f"{role}: label/origin latih >= awal evaluasi {start.date()} -- dataset belum di-purge. "
                           "Jalankan ulang `python run.py preprocess`.")


# ───────────────────────────────────────────── sequence (Phase 2 SEQUENCE_CONTRACT)
class SequenceBank:
    """Sequence per (komoditas, origin): harga (L × 4) dan iklim eksogen (L × n_clim)."""

    def __init__(self, L: int):
        self.L = L
        self.prices = {}
        for c in COMMODITIES:
            df = pd.read_csv(DATA_PROCESSED / "pihps" / f"prices_{c}.csv", parse_dates=["date"])
            self.prices[c] = (df["date"].to_numpy("datetime64[D]"), np.log(df["price"].to_numpy(float)))
        clim = pd.read_csv(DATA_PROCESSED / "climate" / "climate_tier1_mean.csv", parse_dates=["date"])
        self.clim_dates = clim["date"].to_numpy("datetime64[D]")
        self.clim_cols = [c for c in clim.columns if c != "date"]
        self.clim = clim[self.clim_cols].to_numpy(float)

    def price_seq(self, c: str, t: np.datetime64) -> np.ndarray:
        dates, logp = self.prices[c]
        i = int(np.searchsorted(dates, t, side="right"))             # observasi <= t
        j = max(0, i - self.L)
        d, lp = dates[j:i], logp[j:i]
        out = np.zeros((self.L, SEQ_PRICE_CHANNELS), np.float32)
        n = len(d)
        if n == 0:
            return out
        prev_d, prev_lp = (dates[j - 1], logp[j - 1]) if j > 0 else (d[0], lp[0])
        ret = np.diff(np.concatenate([[prev_lp], lp]))
        gap = np.diff(np.concatenate([[prev_d], d])).astype("timedelta64[D]").astype(float)
        out[-n:, 0] = lp - logp[i - 1]
        out[-n:, 1] = ret
        out[-n:, 2] = gap / 7.0
        out[-n:, 3] = 1.0
        return out

    def clim_seq(self, c: str, t: np.datetime64) -> np.ndarray:
        """Iklim pada tanggal (d_i - lag) untuk tiap observasi harga d_i di jendela; NaN -> diisi nanti."""
        dates, _ = self.prices[c]
        i = int(np.searchsorted(dates, t, side="right"))
        d = dates[max(0, i - self.L):i] - np.timedelta64(CLIMATE_LAG, "D")
        out = np.full((self.L, len(self.clim_cols)), np.nan, np.float32)
        if len(d):
            k = np.searchsorted(self.clim_dates, d)
            ok = (k < len(self.clim_dates)) & (self.clim_dates[np.minimum(k, len(self.clim_dates) - 1)] == d)
            vals = np.full((len(d), len(self.clim_cols)), np.nan)
            vals[ok] = self.clim[k[ok]]
            out[-len(d):] = vals
        return out


# ───────────────────────────────────────────── tensor satu split
@dataclass
class Tensors:
    c: np.ndarray                 # indeks komoditas
    hi: np.ndarray                # indeks horizon
    h: np.ndarray
    origin: np.ndarray
    y: np.ndarray                 # r
    price_origin: np.ndarray
    price_label: np.ndarray
    oi: np.ndarray                # indeks ke bank sequence
    x: dict = field(default_factory=dict)        # {kelompok: matriks terstandardisasi}
    keys: pd.DataFrame | None = None

    def __len__(self):
        return len(self.y)


@dataclass
class FoldData:
    role: str
    L: int
    train: Tensors
    val: Tensors
    seq_price: np.ndarray         # (N_origin, L, 4)
    seq_clim: np.ndarray          # (N_origin, L, n_clim) terstandardisasi + kanal mask terakhir
    dims: dict
    frames_train: pd.DataFrame
    frames_val: pd.DataFrame


Z_CLIP = 5.0


class Preprocessor:
    """Imputasi median + standardisasi, di-fit per komoditas di data LATIH saja.

    Nilai terstandardisasi dipotong ±Z_CLIP: kejadian di luar rentang latih (lonjakan berita, shock harga) tetap
    terbaca sebagai ekstrem, tanpa nilai ratusan σ yang membuat model linear/neural mengekstrapolasi liar.
    """

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.stats = {}

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for c, g in df.groupby("comcat_id"):
            X = g[self.cols].to_numpy(float)
            med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xf = np.where(np.isfinite(X), X, med)
            mu, sd = Xf.mean(0), Xf.std(0)
            self.stats[c] = (med, mu, np.where(sd > 1e-12, sd, 1.0))
        return self

    def transform(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        out = np.zeros((len(df), len(self.cols)), np.float32)
        cid = df["comcat_id"].to_numpy()
        for c, (med, mu, sd) in self.stats.items():
            m = cid == c
            X = df.loc[m, self.cols].to_numpy(float)
            X = np.where(np.isfinite(X), X, med)
            out[m] = np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP) if scale else X
        return out


def build_fold(role: str, tr: pd.DataFrame, ev: pd.DataFrame, L: int) -> FoldData:
    bank = SequenceBank(L)
    cidx = {c: i for i, c in enumerate(COMMODITIES)}
    groups = {g: feature_columns(tr, (g,)) for g in GROUPS}
    pre = {g: Preprocessor(cols).fit(tr) for g, cols in groups.items() if cols}

    # bank sequence untuk semua (komoditas, origin) di kedua split
    both = pd.concat([tr[["comcat_id", "origin_date"]], ev[["comcat_id", "origin_date"]]]).drop_duplicates()
    both = both.sort_values(["comcat_id", "origin_date"]).reset_index(drop=True)
    key_index = {(c, t): i for i, (c, t) in enumerate(zip(both["comcat_id"], both["origin_date"].to_numpy("datetime64[D]")))}
    sp = np.stack([bank.price_seq(c, t) for c, t in key_index])
    sc = np.stack([bank.clim_seq(c, t) for c, t in key_index])
    # standardisasi iklim: statistik dari origin LATIH saja
    tr_keys = [key_index[(c, t)] for c, t in zip(tr["comcat_id"], tr["origin_date"].to_numpy("datetime64[D]"))]
    ref = sc[np.unique(tr_keys)].reshape(-1, sc.shape[2])
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    clim_mask = np.isfinite(sc).all(axis=2, keepdims=True).astype(np.float32)
    sc = np.where(np.isfinite(sc), (sc - mu) / sd, 0.0).astype(np.float32)
    sc = np.concatenate([sc, clim_mask], axis=2)

    def tensors(df: pd.DataFrame) -> Tensors:
        o = df["origin_date"].to_numpy("datetime64[D]")
        return Tensors(
            c=df["comcat_id"].map(cidx).to_numpy(int),
            hi=np.searchsorted(HORIZONS, df["h"].to_numpy(int)),
            h=df["h"].to_numpy(int), origin=o, y=df["r"].to_numpy(float),
            price_origin=df["price_origin"].to_numpy(float), price_label=df["price_label"].to_numpy(float),
            oi=np.array([key_index[(c, t)] for c, t in zip(df["comcat_id"], o)]),
            x={g: p.transform(df) for g, p in pre.items()}
              | ({"DRIFT_RAW": np.nan_to_num(df[groups["DRIFT"]].to_numpy(np.float32), nan=0.0)}
                 if groups.get("DRIFT") else {}),
            keys=df[["comcat_id", "h", "origin_date", "label_date"]].reset_index(drop=True),
        )

    return FoldData(role=role, L=L, train=tensors(tr), val=tensors(ev), seq_price=sp, seq_clim=sc,
                    dims={g: len(c) for g, c in groups.items()} | {"CLIM_SEQ": sc.shape[2]},
                    frames_train=tr, frames_val=ev)


_CACHE: dict = {}


def load_fold(role: str, L: int = 60) -> FoldData:
    key = (role, L)
    if key not in _CACHE:
        _CACHE.clear()                                         # satu fold di memori
        _CACHE[key] = build_fold(role, *read_split(role), L)
    return _CACHE[key]


def target_scale(T: Tensors) -> np.ndarray:
    """σ_{c,h} = simpangan baku r di data latih per (komoditas, horizon)."""
    sig = np.ones((len(COMMODITIES), len(HORIZONS)))
    for ci in range(len(COMMODITIES)):
        for hi in range(len(HORIZONS)):
            v = T.y[(T.c == ci) & (T.hi == hi)]
            sig[ci, hi] = max(float(np.std(v)), 1e-4) if len(v) else 1.0
    return sig


def shock_thresholds(T: Tensors, q: float = 0.95) -> np.ndarray:
    """Ambang shock Q_q |r| data latih per (c, h); inf bila tidak informatif (Q = 0)."""
    thr = np.full((len(COMMODITIES), len(HORIZONS)), np.inf)
    for ci in range(len(COMMODITIES)):
        for hi in range(len(HORIZONS)):
            v = np.abs(T.y[(T.c == ci) & (T.hi == hi)])
            if len(v) and np.quantile(v, q) > 0:
                thr[ci, hi] = np.quantile(v, q)
    return thr
