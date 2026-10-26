"""
ARIF-Net — Model tabular: Ridge, SVR, Random Forest, XGBoost, LightGBM (baseline linear/kernel/pohon).

Satu model per komoditas, pooled atas 8 horizon dengan horizon sebagai fitur (one-hot untuk Ridge/SVR; h & log h
untuk pohon). Target distandardisasi σ_{c,h} dari data latih. Imputasi median & standardisasi fitur di-fit di data
LATIH per komoditas (data.Preprocessor); Ridge/SVR memakai fitur terstandardisasi (wajib untuk model berbasis jarak
/ penalti), pohon memakai nilai asli terimputasi.
Set fitur: "harga" (kelompok PRICE) atau "multimodal" (PRICE + CLIM + NEWS + BBM + CAL).
"""
from __future__ import annotations

import logging

import numpy as np

from ..data import (COMMODITIES, HORIZONS, MULTIMODAL, FoldData, Preprocessor, feature_columns, resolve_threads,
                    target_scale)

N_JOBS = resolve_threads()          # core fisik mesin (ARIFNET_THREADS untuk menimpa)
SCALED = ("ridge", "svr")


def make(kind: str, p: dict, seed: int):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=p.get("alpha", 1.0))
    if kind == "svr":
        from sklearn.svm import SVR
        return SVR(C=p.get("C", 1.0), epsilon=p.get("epsilon", 0.1), gamma=p.get("gamma", "scale"),
                   kernel=p.get("kernel", "rbf"), cache_size=1000)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
                                     min_samples_leaf=p.get("min_samples_leaf", 5),
                                     max_features=p.get("max_features", 0.3), n_jobs=N_JOBS, random_state=seed)
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
                            learning_rate=p.get("learning_rate", 0.03), subsample=p.get("subsample", 0.8),
                            colsample_bytree=p.get("colsample_bytree", 0.5),
                            min_child_weight=p.get("min_child_weight", 5), reg_lambda=p.get("reg_lambda", 1.0),
                            objective=p.get("objective", "reg:pseudohubererror"),
                            n_jobs=N_JOBS, random_state=seed, verbosity=0)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=p.get("n_estimators", 400), num_leaves=p.get("num_leaves", 15),
                             learning_rate=p.get("learning_rate", 0.03), subsample=p.get("subsample", 0.8),
                             subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.5),
                             min_child_samples=p.get("min_child_samples", 30), reg_lambda=p.get("reg_lambda", 1.0),
                             objective=p.get("objective", "huber"), alpha=p.get("huber_alpha", 1.0),
                             n_jobs=N_JOBS, random_state=seed, verbose=-1)
    raise KeyError(f"model tabular tidak dikenal: {kind}")


def horizon_block(h: np.ndarray, one_hot: bool) -> np.ndarray:
    if one_hot:
        return (h[:, None] == np.array(HORIZONS)[None, :]).astype(np.float32)
    return np.column_stack([h, np.log(h)]).astype(np.float32)


def fit_predict(fd: FoldData, cfg: dict, seed: int, log=print, extra: dict | None = None) -> dict:
    """extra = {"train": matriks, "val": matriks} kolom tambahan (mis. embedding TCN) yang digabung ke fitur."""
    m = cfg["model"]
    kind, p = m["kind"], m.get("params", {})
    groups = ("PRICE",) if m.get("features", "multimodal") == "harga" else MULTIMODAL
    scale = kind in SCALED
    tr, va = fd.frames_train, fd.frames_val
    cols = feature_columns(tr, groups)
    pre = Preprocessor(cols).fit(tr)
    Xtr_all, Xva_all = pre.transform(tr, scale=scale), pre.transform(va, scale=scale)
    sig = target_scale(fd.train)
    pred = np.zeros(len(va))
    for ci, c in enumerate(COMMODITIES):
        mtr, mva = fd.train.c == ci, fd.val.c == ci
        parts_tr = [Xtr_all[mtr], horizon_block(fd.train.h[mtr], scale)]
        parts_va = [Xva_all[mva], horizon_block(fd.val.h[mva], scale)]
        if extra is not None:
            parts_tr.append(extra["train"][mtr])
            parts_va.append(extra["val"][mva])
        Xtr, Xva = np.hstack(parts_tr), np.hstack(parts_va)
        s_tr = sig[ci, fd.train.hi[mtr]]
        est = make(kind, p, seed)
        est.fit(Xtr, fd.train.y[mtr] / s_tr)
        pred[mva] = est.predict(Xva) * sig[ci, fd.val.hi[mva]]
        log(f"    {c}: {kind} {Xtr.shape[0]} × {Xtr.shape[1]} fitur")
    logging.getLogger(__name__).debug("tabular %s selesai", kind)
    return {"r_hat": pred}
