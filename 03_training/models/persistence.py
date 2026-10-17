"""
ARIF-Net — Persistence baseline: r̂ = 0 (harga tidak berubah).

Naive forecast yang menjadi lower bound untuk semua model.
"""
from __future__ import annotations

import numpy as np


class PersistenceModel:
    """Prediksi r̂ = 0 untuk semua observasi."""

    name = "persistence"

    def fit(self, X_train, y_train, **kwargs):
        pass  # No-op

    def predict(self, X_val) -> np.ndarray:
        return np.zeros(len(X_val))

    def __repr__(self):
        return "PersistenceModel(r̂ = 0)"
