"""
ARIF-Net — ARIMA baseline: univariat pada riwayat harga per komoditas.

ARIMA difit per komoditas pada harga (bukan return), kemudian prediksi
dihitung sebagai return r = (P̂(t+h) − P(t)) / P(t).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd


class ARIMAModel:
    """ARIMA univariat sebagai baseline statistik klasik."""

    name = "arima"

    def __init__(self, order=(5, 1, 1), seasonal=False):
        self.order = tuple(order)
        self.seasonal = seasonal
        self.models = {}

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **kwargs):
        """Fit ARIMA per komoditas pada data harga historis."""
        from statsmodels.tsa.arima.model import ARIMA

        for cid in X_train["comcat_id"].unique():
            mask = X_train["comcat_id"] == cid
            prices = X_train.loc[mask, "price_origin"].values
            try:
                model = ARIMA(prices, order=self.order)
                self.models[cid] = model.fit(method_kwargs={"maxiter": 200})
                logging.info("ARIMA fit %s: AIC=%.1f", cid, self.models[cid].aic)
            except Exception as exc:
                logging.warning("ARIMA fit gagal %s: %s", cid, exc)
                self.models[cid] = None

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        """Prediksi return r̂ = (P̂(t+h) − P(t)) / P(t)."""
        preds = np.zeros(len(X_val))
        for cid in X_val["comcat_id"].unique():
            mask = X_val["comcat_id"] == cid
            model = self.models.get(cid)
            if model is None:
                continue  # Fallback to 0 (persistence)
            p0 = X_val.loc[mask, "price_origin"].values
            h = X_val.loc[mask, "h"].values
            # Forecast using trained model
            try:
                forecast = model.forecast(steps=max(h))
                for i, (p, horizon) in enumerate(zip(p0, h)):
                    if horizon <= len(forecast) and p > 0:
                        preds[mask.values.nonzero()[0][i]] = (forecast[horizon - 1] - p) / p
            except Exception:
                pass  # Fallback to 0
        return preds

    def __repr__(self):
        return f"ARIMAModel(order={self.order})"
