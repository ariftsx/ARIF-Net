"""
ARIF-Net — Baseline ARIMA (Manogna et al., 2025; Amalia et al., 2026; PROTOKOL_EKSPERIMEN P4-DG-09).

Per komoditas, pada log harga seri hari terlapor (indeks hari kerja):
  1. d dipilih dengan uji ADF (α = 0,05) pada data latih: stasioner → d = 0, bila tidak → d = 1 (Manogna, Alg. ARIMA).
  2. (p, q) dipilih dengan AIC dari grid kecil pada data latih (setara auto_arima, Amalia et al.).
  3. Parameter di-fit SEKALI per fold pada seri < batas evaluasi; untuk tiap origin t model difilter maju sampai t
     (hanya observasi ≤ t, parameter tetap) lalu diramal k langkah, k = jumlah hari kerja di (t, t+h].
  r̂ = exp(ln P̂(t+k) − ln P(t)) − 1.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data import COMMODITIES, DATA_PROCESSED, FoldData


def business_steps(t: np.datetime64, h: int) -> int:
    return max(int(np.busday_count(t + np.timedelta64(1, "D"), t + np.timedelta64(h + 1, "D"))), 1)


def choose_d(x: np.ndarray, alpha: float = 0.05) -> int:
    from statsmodels.tsa.stattools import adfuller
    return 0 if adfuller(x, autolag="AIC")[1] < alpha else 1


def fit_predict(fd: FoldData, cfg: dict, seed: int, log=print) -> dict:
    from statsmodels.tsa.arima.model import ARIMA
    m = cfg["model"]
    p_grid, q_grid = m.get("p", [0, 1, 2, 3, 5]), m.get("q", [0, 1, 2])
    pred = np.zeros(len(fd.val))
    info = {}
    for ci, c in enumerate(COMMODITIES):
        df = pd.read_csv(DATA_PROCESSED / "pihps" / f"prices_{c}.csv", parse_dates=["date"])
        dates, logp = df["date"].to_numpy("datetime64[D]"), np.log(df["price"].to_numpy(float))
        mva = fd.val.c == ci
        o_va, h_va = fd.val.origin[mva], fd.val.h[mva]
        n_tr = int(np.searchsorted(dates, o_va.min(), side="left"))     # seri latih: < awal evaluasi
        d = m.get("d") if m.get("d") is not None else choose_d(logp[:n_tr])
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in p_grid:
                for q in q_grid:
                    try:
                        res = ARIMA(logp[:n_tr], order=(p, d, q), trend="n").fit()
                    except Exception:                                    # noqa: BLE001 — ordo gagal konvergen dilewati
                        continue
                    if np.isfinite(res.aic) and (best is None or res.aic < best[1].aic):
                        best = ((p, d, q), res)
        if best is None:
            raise RuntimeError(f"ARIMA {c}: tidak ada ordo yang konvergen")
        order, res = best
        cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in np.unique(o_va):
                i = int(np.searchsorted(dates, t, side="right"))         # observasi ≤ t
                rf = res.apply(logp[:i])                                 # filter maju, parameter tetap
                kmax = max(business_steps(t, h) for h in h_va[o_va == t])
                cache[t] = (np.asarray(rf.forecast(kmax)), logp[i - 1])
        out = np.empty(len(o_va))
        for j, (t, h) in enumerate(zip(o_va, h_va)):
            fc, last = cache[t]
            out[j] = np.exp(fc[business_steps(t, h) - 1] - last) - 1
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"ARIMA {c}: prediksi tidak hingga")
        pred[mva] = out
        info[c] = {"order": list(order), "aic": float(res.aic)}
        log(f"    {c}: ARIMA{order} AIC {res.aic:.1f}; {len(np.unique(o_va))} origin")
    return {"r_hat": pred, "info": info}
