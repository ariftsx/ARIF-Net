"""
ARIF-Net — Metrik evaluasi regresi untuk target r(t,h).

Metrik: MAE, RMSE, relMAE, Directional Accuracy, MAPE, R², Diebold-Mariano test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


COMMODITIES = ("com_14", "com_11", "com_3")
COMMODITY_NAMES = {
    "com_14": "Cabai Merah Keriting",
    "com_11": "Bawang Merah Ukuran Sedang",
    "com_3": "Beras Kualitas Medium I",
}


def basic(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrik dasar pada target r."""
    nz = y != 0
    mae = float(np.mean(np.abs(y - p)))
    mae0 = float(np.mean(np.abs(y)))  # MAE persistence (r̂=0)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    rmse0 = float(np.sqrt(np.mean(y ** 2)))
    return {
        "n": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "mae_persistence": mae0,
        "rmse_persistence": rmse0,
        "relmae": mae / mae0 if mae0 > 0 else None,
        "relrmse": rmse / rmse0 if rmse0 > 0 else None,
        "dir_acc": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else None,
        "r_squared": float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
            if np.var(y) > 0 else None,
    }


def mape_price(price_origin: np.ndarray, price_label: np.ndarray,
               r_hat: np.ndarray) -> float:
    """MAPE pada harga rekonstruksi P̂ = P(t) * (1 + r̂)."""
    p_hat = price_origin * (1 + r_hat)
    mask = price_label > 0
    return float(np.mean(np.abs((price_label[mask] - p_hat[mask]) / price_label[mask]))) * 100


def table(df: pd.DataFrame, pred_col: str = "r_hat") -> pd.DataFrame:
    """Tabel metrik per (komoditas, horizon)."""
    rows = []
    for (c, h), g in df.groupby(["comcat_id", "h"], sort=False):
        y, p = g["r"].to_numpy(), g[pred_col].to_numpy()
        row = {"komoditas": c, "h": int(h), **basic(y, p)}
        if "price_origin" in g.columns and "price_label" in g.columns:
            row["mape_price"] = mape_price(g["price_origin"].values,
                                           g["price_label"].values, p)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["komoditas", "h"]).reset_index(drop=True)


def overall(tab: pd.DataFrame) -> dict:
    """Metrik gabungan berbobot jumlah observasi."""
    w = tab["n"]
    return {
        "relmae": float((tab["mae"] * w).sum() / (tab["mae_persistence"] * w).sum()),
        "mae": float((tab["mae"] * w).sum() / w.sum()),
        "rmse": float(np.sqrt((tab["rmse"] ** 2 * w).sum() / w.sum())),
        "dir_acc": float((tab["dir_acc"].fillna(0) * w).sum() / w.sum()),
        "n": int(w.sum()),
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Uji Diebold-Mariano: H₀ = akurasi kedua model sama.

    e1, e2 = error (y - ŷ) dari model 1 dan model 2.
    h = horizon (untuk koreksi autokorelasi).
    Return: DM statistic, p-value, dan interpretation.
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = np.mean(d)
    # HAC variance (Newey-West dengan h-1 lag)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant": False}
    dm = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {
        "dm_stat": float(dm),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "better": "model1" if dm > 0 else "model2",
    }
