"""
ARIF-Net — uji unit (tanpa data model; jalankan: python -m 04_arif_net.selftest).

  1. mixture_median: tiga kasus SPEC §2.5 + kecocokan dengan median Monte Carlo campuran
  2. kepala kuantil monoton untuk input acak
  3. fitur drift tidak bocor: mengubah harga SESUDAH t tidak mengubah fitur pada origin t
"""
from __future__ import annotations

import sys
from importlib import import_module

import numpy as np
import pandas as pd
import torch

from .model import TAUS, ARIFNet, mixture_median


def test_mixture_median() -> None:
    q_neg = np.array([[-0.10, -0.08, -0.05, -0.03, -0.02]])
    q_pos = -q_neg[:, ::-1]
    assert mixture_median(np.array([0.4]), q_pos)[0] == 0.0                       # p ≤ 0,5 → 0
    m = mixture_median(np.array([0.9]), q_neg)[0]                                  # a ≥ 0,5 → negatif
    assert np.isclose(m, np.interp(0.5 / 0.9, TAUS, q_neg[0])) and m < 0
    m = mixture_median(np.array([0.9]), q_pos)[0]                                  # positif
    assert np.isclose(m, np.interp(0.4 / 0.9, TAUS, q_pos[0])) and m > 0
    rng = np.random.default_rng(0)
    for _ in range(200):                                                           # Monte Carlo
        p = rng.uniform(0.05, 1.0)
        q = np.sort(rng.normal(rng.normal(0, 0.05), 0.05, 5))
        u = rng.uniform(size=200_000)
        changed = rng.uniform(size=u.size) < p
        g = np.interp(np.clip(u, 0.1, 0.9), TAUS, q)                             # G pada [τ_min, τ_max]
        x = np.where(changed, g, 0.0)
        assert abs(mixture_median(np.array([p]), q[None])[0] - np.median(x)) < 5e-3, (p, q)


def test_monotone() -> None:
    dims = {"PRICE": 5, "CAL": 3, "CALF": 3}
    net = ARIFNet(dims, d=16, drift=False)
    for p in net.q_head.parameters():
        torch.nn.init.normal_(p, std=2.0)
    n = 64
    B = {"seq": torch.randn(n, 10, 4), "c": torch.randint(0, 3, (n,)), "hi": torch.randint(0, 8, (n,)),
         "x_PRICE": torch.randn(n, 5), "x_CAL": torch.randn(n, 3), "x_CALF": torch.randn(n, 3)}
    q = net(B, torch.ones(n))["q"]
    assert torch.all(q[:, 1:] >= q[:, :-1])


def test_drift_no_leak() -> None:
    bd = import_module("02_preprocessing.build_dataset")
    dates = pd.bdate_range("2019-01-01", "2023-12-29")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({"date": dates, "price": 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))})
    t = np.datetime64("2022-06-15")
    o, h = np.array([t] * 3), np.array([30, 180, 365])
    a = bd.build_drift_features(prices, o, h)
    fut = prices.copy()
    fut.loc[fut["date"] > pd.Timestamp(t), "price"] *= 3.0                          # ubah masa depan
    b = bd.build_drift_features(fut, o, h)
    pd.testing.assert_frame_equal(a, b)


def main() -> int:
    for f in (test_mixture_median, test_monotone, test_drift_no_leak):
        f()
        print(f"LULUS {f.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
