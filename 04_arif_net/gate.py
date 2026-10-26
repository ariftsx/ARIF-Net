"""
ARIF-Net — ablation X-1 / G-4: gerbang expert (docs/ARIFNET_SPEC.md §2.6).

Expert: ARIF-Net inti (konfigurasi beku A0), LightGBM multimodal (konfigurasi beku baseline), persistence.
Bobot w_{c,h} pada simpleks (grid 0,1) dipilih per (komoditas, horizon) dengan loss absolut pada prediksi
OUT-OF-FOLD di DALAM data latih: origin latih dipotong pada kuantil 55/70/85%; tiap potongan k, expert dilatih
pada origin & label < c_k lalu memprediksi c_k ≤ origin < c_{k+1} (purging setara SPLIT_SPEC). Lalu expert
dilatih ulang pada seluruh data latih fold dan digabung dengan bobot tersebut. Data evaluasi tidak dipakai.
"""
from __future__ import annotations

import itertools
import json
from importlib import import_module

import numpy as np
import pandas as pd

_data = import_module("03_training.data")

CUTS = (0.55, 0.70, 0.85)
GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=2) if w[0] + w[1] <= 1.0 + 1e-9]


def _frozen(model: str) -> dict:
    p = _data.RESULTS / "frozen" / f"{model}.json"
    if not p.exists():
        raise FileNotFoundError(f"gerbang butuh konfigurasi beku {model} ({p})")
    return json.loads(p.read_text(encoding="utf-8"))["config"]


def _experts(fd, cfgs: dict, seed: int, log) -> np.ndarray:
    runs = import_module("03_training.runs")
    cols = []
    for name, cfg in cfgs.items():
        out, _ = runs.predict_with(name, fd, cfg, seed, log=lambda *_: None)
        cols.append(out["r_hat"])
    return np.column_stack(cols)                                    # (n, 2): arifnet, lgbm; persistence = 0


def fit_predict(fd, cfg: dict, seed: int, log=print) -> tuple[dict, dict]:
    cfgs = {"arifnet": _frozen("arifnet"), "lgbm": _frozen("lgbm")}
    L = int(cfgs["arifnet"]["train"]["L"])
    tr = fd.frames_train
    days = np.sort(tr["origin_date"].unique())
    cut = [pd.Timestamp(days[int(q * (len(days) - 1))]) for q in CUTS] + [None]
    oof = []
    for k in range(len(CUTS)):
        a, b = cut[k], cut[k + 1]
        sub_tr = tr[(tr["origin_date"] < a) & (tr["label_date"] < a)]
        sub_ev = tr[(tr["origin_date"] >= a) & ((tr["origin_date"] < b) if b is not None else True)]
        sub = _data.build_fold(f"{fd.role}-gate{k}", sub_tr, sub_ev, L)
        P = _experts(sub, cfgs, seed, log)
        oof.append(pd.DataFrame({"c": sub.val.c, "hi": sub.val.hi, "y": sub.val.y, "a": P[:, 0], "g": P[:, 1]}))
        log(f"    gerbang OOF potongan {k + 1}/{len(CUTS)}: {len(sub_tr)} latih → {len(sub_ev)} OOF")
    O = pd.concat(oof, ignore_index=True)
    W = {}
    for (c, hi), g in O.groupby(["c", "hi"]):
        losses = [np.abs(g.y - (w1 * g.a + w2 * g.g)).sum() for w1, w2 in GRID]
        W[(c, hi)] = GRID[int(np.argmin(losses))]
    P = _experts(fd, cfgs, seed, log)
    w = np.array([W.get((c, hi), (0.0, 0.0)) for c, hi in zip(fd.val.c, fd.val.hi)])
    r_hat = w[:, 0] * P[:, 0] + w[:, 1] * P[:, 1]
    weights = {f"{_data.COMMODITIES[c]}|h{_data.HORIZONS[hi]}": {"arifnet": v[0], "lgbm": v[1],
                                                                 "persistence": round(1 - v[0] - v[1], 1)}
               for (c, hi), v in W.items()}
    return {"r_hat": r_hat}, {"gate_weights": weights}
