"""
ARIF-Net — Model tabular: Ridge, SVR, Random Forest, XGBoost, LightGBM.

Satu model per komoditas (kolom iklim/logistik berbeda per komoditas), pooled atas 8 horizon
dengan horizon sebagai fitur. Target distandardisasi σ_{c,h} dari data latih.
Set fitur: "harga" (modality Price) atau "multimodal" (semua kolom).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 7, 14, 30, 90, 180, 365)
N_JOBS = 2


def _feature_columns(df: pd.DataFrame, features: str = "multimodal") -> list[str]:
    """Kolom fitur berdasarkan mode."""
    meta_cols = {"comcat_id", "commodity", "origin_date", "h",
                 "target_date_nominal", "label_date", "label_gap_days",
                 "price_origin", "price_label", "r",
                 "origin_holiday_carry", "origin_spike_reversal"}
    if features == "harga":
        return [c for c in df.columns if c.startswith(("price_", "return_", "volatility_", "momentum_"))
                and c not in meta_cols]
    return [c for c in df.columns if c not in meta_cols and c != "r"]


def _design(df: pd.DataFrame, cols: list[str], one_hot: bool) -> np.ndarray:
    """Bangun matriks fitur dengan horizon encoding."""
    X = df[cols].fillna(0).to_numpy(np.float32)
    h = df["h"].to_numpy()
    if one_hot:
        H = (h[:, None] == np.array(HORIZONS)[None, :]).astype(np.float32)
        return np.hstack([X, H])
    return np.hstack([X, h[:, None].astype(np.float32), np.log(h)[:, None].astype(np.float32)])


def _sigma_ch(df: pd.DataFrame) -> dict:
    """σ_{c,h} target standardization dari data latih."""
    return {(c, h): max(float(g["r"].std(ddof=0)), 1e-4)
            for (c, h), g in df.groupby(["comcat_id", "h"])}


def _make(kind: str, params: dict, seed: int):
    """Buat instance estimator sklearn."""
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params.get("alpha", 1.0))
    if kind == "svr":
        from sklearn.svm import SVR
        return SVR(C=params.get("C", 1.0), epsilon=params.get("epsilon", 0.1),
                   gamma=params.get("gamma", "scale"), cache_size=1000)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", 0.3),
            n_jobs=N_JOBS, random_state=seed)
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.5),
            min_child_weight=params.get("min_child_weight", 5),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="reg:pseudohubererror",
            n_jobs=N_JOBS, random_state=seed, verbosity=0)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=params.get("n_estimators", 400),
            num_leaves=params.get("num_leaves", 15),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.8), subsample_freq=1,
            colsample_bytree=params.get("colsample_bytree", 0.5),
            min_child_samples=params.get("min_child_samples", 30),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="huber", alpha=params.get("huber_alpha", 1.0),
            n_jobs=N_JOBS, random_state=seed, verbose=-1)
    raise KeyError(f"Unknown tabular kind: {kind}")


class TabularModel:
    """Model tabular sklearn-compatible untuk regresi return harga."""

    def __init__(self, kind: str = "ridge", features: str = "multimodal",
                 params: dict | None = None, seed: int = 0):
        self.kind = kind
        self.features = features
        self.params = params or {}
        self.seed = seed
        self.name = f"{kind}_{features}"
        self.estimators = {}
        self.sigmas = {}
        self.cols = {}

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **kwargs):
        """Fit per komoditas (pooled over horizons)."""
        one_hot = self.kind in ("ridge", "svr")
        for cid in X_train["comcat_id"].unique():
            mask = X_train["comcat_id"] == cid
            tr = X_train[mask]
            cols = _feature_columns(tr, self.features)
            self.cols[cid] = cols
            X = _design(tr, cols, one_hot)
            sigma = _sigma_ch(tr)
            self.sigmas[cid] = sigma
            # Standardize target
            y_std = np.array([tr.iloc[i]["r"] / sigma.get((cid, tr.iloc[i]["h"]), 1e-4)
                              for i in range(len(tr))], dtype=np.float32)
            est = _make(self.kind, self.params, self.seed)
            est.fit(X, y_std)
            self.estimators[cid] = est
            logging.info("  %s: %s pada %d × %d fitur", cid, self.kind, X.shape[0], X.shape[1])

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        """Prediksi return."""
        one_hot = self.kind in ("ridge", "svr")
        preds = np.zeros(len(X_val))
        for cid in X_val["comcat_id"].unique():
            mask = X_val["comcat_id"] == cid
            va = X_val[mask]
            cols = self.cols.get(cid, _feature_columns(va, self.features))
            X = _design(va, cols, one_hot)
            est = self.estimators.get(cid)
            if est is None:
                continue
            y_std = est.predict(X)
            # Un-standardize
            sigma = self.sigmas.get(cid, {})
            for i, idx in enumerate(mask.values.nonzero()[0]):
                h = X_val.iloc[idx]["h"]
                s = sigma.get((cid, h), 1e-4)
                preds[idx] = y_std[i] * s
        return preds

    def __repr__(self):
        return f"TabularModel(kind={self.kind}, features={self.features})"
