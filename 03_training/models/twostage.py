"""
ARIF-Net — TCN-XGBoost hybrid (Zhao et al., 2025).

Tahap 1: TCN (6 layer, kernel 3, dilasi 1…32, dropout 0,3, WeightNorm) DILATIH end-to-end pada r(t,h) atas window
         harga 30 hari (MSE, Adam).
Tahap 2: representasi TCN (keluaran langkah terakhir + embedding c, h) → XGBoost per komoditas (default Tabel 5:
         max_depth 10, learning_rate 0,03, subsample 1,0, reg_lambda 1,2; loss kuadrat). Opsi stage2.features =
         "multimodal" menambah fitur tabular multimodal (varian protokol lama), default "embedding" = sesuai paper.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import trainer
from . import tabular
from .neural import build_neural


def fit_predict(fd, cfg: dict, seed: int, log=print) -> tuple[dict, dict]:
    s1 = {"model": cfg["stage1"]["model"], "train": cfg["stage1"]["train"]}
    torch.manual_seed(seed)
    net = build_neural("tcn", s1, fd)
    log(f"  tahap 1: TCN ({trainer.n_params(net):,} parameter), {s1['train']['epochs']} epoch")
    info = trainer.fit(net, fd, s1, seed, log=log)
    e_tr = trainer.embed(net, fd, fd.train, info["sigma"])
    e_va = trainer.embed(net, fd, fd.val, info["sigma"])
    feats = cfg["stage2"].get("features", "embedding")
    log(f"  tahap 2: XGBoost atas embedding TCN{' + fitur multimodal' if feats == 'multimodal' else ''}")
    s2 = {"model": {"kind": "xgb", "features": "multimodal", "params": cfg["stage2"].get("params", {})}}
    if feats == "multimodal":
        out = tabular.fit_predict(fd, s2, seed, log=log, extra={"train": e_tr, "val": e_va})
    else:
        out = _xgb_embedding_only(fd, s2["model"]["params"], seed, e_tr, e_va, log)
    return out, {**info, "n_params": trainer.n_params(net)}


def _xgb_embedding_only(fd, params: dict, seed: int, e_tr, e_va, log) -> dict:
    from ..data import COMMODITIES, target_scale
    sig = target_scale(fd.train)
    pred = np.zeros(len(fd.val))
    for ci, c in enumerate(COMMODITIES):
        mtr, mva = fd.train.c == ci, fd.val.c == ci
        Xtr = np.hstack([e_tr[mtr], tabular.horizon_block(fd.train.h[mtr], False)])
        Xva = np.hstack([e_va[mva], tabular.horizon_block(fd.val.h[mva], False)])
        est = tabular.make("xgb", params, seed)
        est.fit(Xtr, fd.train.y[mtr] / sig[ci, fd.train.hi[mtr]])
        pred[mva] = est.predict(Xva) * sig[ci, fd.val.hi[mva]]
        log(f"    {c}: XGB {Xtr.shape[0]} × {Xtr.shape[1]}")
    return {"r_hat": pred}
