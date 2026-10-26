"""
ARIF-Net — Metrik evaluasi regresi untuk target r(t,h) (PROTOKOL_EKSPERIMEN §5).

  Pada r     : MAE, RMSE, relMAE (penyebut persistence periode evaluasi = metrik seleksi), MASE (penyebut persistence
               data LATIH per (c, h)), R², Directional Accuracy (hanya r ≠ 0).
  Pada harga : MAE/RMSE Rp dan MAPE harga rekonstruksi P̂ = P(t)·(1 + r̂).
  Shock      : MAE pada |r| ≥ Q95 data latih per (c, h) (hanya pasangan informatif).
  Uji        : Diebold–Mariano dengan koreksi Harvey–Leybourne–Newbold (HLN), loss galat absolut, HAC lag k−1 dengan
               k = jumlah hari kerja dalam horizon (label tumpang tindih antar origin berurutan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def relmae(y: np.ndarray, p: np.ndarray) -> float:
    """Σ|r − r̂| / Σ|r − 0| pada seluruh baris (metrik seleksi, P4-DG-05a)."""
    d = float(np.abs(y).sum())
    return float(np.abs(y - p).sum() / d) if d > 0 else float("nan")


def basic(y: np.ndarray, p: np.ndarray, mae_insample: float | None = None) -> dict:
    nz = y != 0
    mae = float(np.mean(np.abs(y - p)))
    mae0 = float(np.mean(np.abs(y)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {
        "n": int(len(y)),
        "mae": mae, "rmse": rmse,
        "mae_persistence": mae0, "rmse_persistence": float(np.sqrt(np.mean(y ** 2))),
        "relmae": mae / mae0 if mae0 > 0 else np.nan,
        "mase": mae / mae_insample if mae_insample else np.nan,
        "dir_acc": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
        "r_squared": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else np.nan,
    }


def price_metrics(price_origin, price_label, r_hat) -> dict:
    p_hat = price_origin * (1 + r_hat)
    m = price_label > 0
    e = price_label[m] - p_hat[m]
    return {"mae_rp": float(np.mean(np.abs(e))), "rmse_rp": float(np.sqrt(np.mean(e ** 2))),
            "mape_price": float(np.mean(np.abs(e / price_label[m])) * 100)}


def table(df: pd.DataFrame, insample: dict | None = None, shock: dict | None = None,
          pred_col: str = "r_hat") -> pd.DataFrame:
    """Metrik per (komoditas, horizon). insample = {(c, h): MAE persistence data latih}; shock = {(c, h): Q95}."""
    rows = []
    for (c, h), g in df.groupby(["comcat_id", "h"], sort=True):
        y, p = g["r"].to_numpy(float), g[pred_col].to_numpy(float)
        row = {"komoditas": c, "h": int(h), **basic(y, p, (insample or {}).get((c, int(h))))}
        if {"price_origin", "price_label"} <= set(g.columns):
            row |= price_metrics(g["price_origin"].to_numpy(float), g["price_label"].to_numpy(float), p)
        thr = (shock or {}).get((c, int(h)), np.inf)
        s = np.abs(y) >= thr
        row["shock_n"] = int(s.sum())
        row["shock_mae"] = float(np.mean(np.abs(y[s] - p[s]))) if s.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def overall(df: pd.DataFrame, pred_col: str = "r_hat") -> dict:
    """Metrik gabungan semua baris (c, h); relMAE = metrik seleksi."""
    y, p = df["r"].to_numpy(float), df[pred_col].to_numpy(float)
    b = basic(y, p)
    return {"relmae": b["relmae"], "mae": b["mae"], "rmse": b["rmse"], "dir_acc": b["dir_acc"],
            "r_squared": b["r_squared"], "n": b["n"]}


def business_days(h: int) -> int:
    """Perkiraan jumlah hari kerja dalam h hari kalender (≥ 1)."""
    return max(1, int(round(h * 5 / 7)))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "abs") -> dict:
    """DM-HLN. H₀: akurasi sama. d = L(e1) − L(e2); dm < 0 → model 1 lebih akurat.

    e1, e2 = galat (r − r̂) model 1 dan model 2, berurutan menurut origin. h = horizon kalender (hari).
    """
    L = np.abs if loss == "abs" else np.square
    d = L(e1) - L(e2)
    n = len(d)
    k = business_days(h)
    d_bar = d.mean()
    gamma = [np.mean((d[j:] - d_bar) * (d[:n - j] - d_bar)) for j in range(0, min(k, n))]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if not np.isfinite(var) or var <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant": False, "better": "seri", "n": n}
    dm = d_bar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n) if n > 2 * k else 1.0
    stat = dm * hln
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {"dm_stat": float(stat), "p_value": p, "significant": p < 0.05,
            "better": ("model1" if stat < 0 else "model2") if p < 0.05 else "seri", "n": n}
