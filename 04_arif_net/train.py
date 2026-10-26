"""
ARIF-Net v1 — training (docs/ARIFNET_SPEC.md §2–§4).

Data, split, preprocessing, σ_{c,h}, ambang shock, dan aturan epoch IDENTIK dengan baseline neural
(03_training/data.py, 03_training/trainer.py): jumlah epoch = hyperparameter dari kurva checkpoint F1–F2.

Loss (target r/σ_{c,h}):
  hurdle (default) : event_weight · BCE(logit, 1[r ≠ 0])  +  pinball(Q_τ, r) pada baris r ≠ 0
  C-1 tanpa hurdle : pinball(Q_τ, r) pada semua baris (median langsung)
  C-3 tanpa kuantil: seperti hurdle, kuantil tunggal τ = 0,5
  C-4 titik MSE    : MSE(r̂, r) pada semua baris (kontrol ketidakcocokan loss vs metrik)
  bobot shock w untuk |r| ≥ Q95 latih per (c, h) pada suku besaran (C-5: w = 1)
"""
from __future__ import annotations

import time
from importlib import import_module

import numpy as np
import torch
from torch import nn

from .model import ARIFNet, point_prediction

_data = import_module("03_training.data")
_trainer = import_module("03_training.trainer")
_metrics = import_module("03_training.metrics")


def build(cfg: dict, fd) -> ARIFNet:
    m = {k: v for k, v in cfg["model"].items() if k not in ("type", "name")}
    return ARIFNet(dims=fd.dims, **m)


def pinball(q: torch.Tensor, y: torch.Tensor, taus) -> torch.Tensor:
    t = torch.as_tensor(taus, dtype=torch.float32)[None, :]
    e = y[:, None] - q
    return torch.maximum(t * e, (t - 1) * e).mean(1)


@torch.no_grad()
def predict(model: ARIFNet, fd, T, sig: np.ndarray, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(T), bs):
        idx = np.arange(i, min(i + bs, len(T)))
        B = _trainer.to_batch(fd, T, idx, sig, model.needs)
        s = torch.as_tensor(sig[T.c[idx], T.hi[idx]], dtype=torch.float32)
        out.append(point_prediction(model, model(B, s)) * sig[T.c[idx], T.hi[idx]])
    return np.concatenate(out) if out else np.zeros(0)


def fit(model: ARIFNet, fd, cfg: dict, seed: int, log=print, curve: bool = False) -> dict:
    tc = cfg["train"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(_data.resolve_threads(tc.get("threads", "auto")))
    T = fd.train
    sig = _data.target_scale(T)
    thr = _data.shock_thresholds(T)
    sw = float(tc.get("shock_weight", 1.0))
    ew = float(tc.get("event_weight", 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    batch = int(tc["batch"])
    epochs = max(_trainer.CHECKPOINT_EPOCHS) if curve else int(tc["epochs"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    curve_rows, hist = [], []
    t0 = time.time()

    def checkpoint(ep):
        if curve and ep in _trainer.CHECKPOINT_EPOCHS:
            curve_rows.append({"epoch": ep, "relmae": _metrics.relmae(fd.val.y, predict(model, fd, fd.val, sig))})

    checkpoint(0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(T))
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            B = _trainer.to_batch(fd, T, idx, sig, model.needs)
            s = torch.as_tensor(sig[T.c[idx], T.hi[idx]], dtype=torch.float32)
            out = model(B, s)
            y = B["y"]
            w = torch.as_tensor(np.where(np.abs(T.y[idx]) >= thr[T.c[idx], T.hi[idx]], sw, 1.0), dtype=torch.float32)
            if model.point:
                loss = ((out["q"][:, 0] - y) ** 2 * w).sum() / w.sum()
            elif model.hurdle:
                nz = (y != 0).float()
                mag = (pinball(out["q"], y, model.taus) * w * nz).sum() / torch.clamp((w * nz).sum(), min=1.0)
                loss = ew * bce(out["logit"], nz).mean() + mag
            else:
                loss = (pinball(out["q"], y, model.taus) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            if tc.get("clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
            losses.append(float(loss.detach()))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)})
        checkpoint(ep)
        if ep in _trainer.CHECKPOINT_EPOCHS or ep == epochs:
            log(f"    epoch {ep:3d}  loss latih {hist[-1]['train_loss']:.4f}"
                + (f"  relMAE evaluasi {curve_rows[-1]['relmae']:.4f}" if curve_rows else ""))
    lam = torch.sigmoid(model.lam).detach().numpy()
    return {"sigma": sig, "history": hist, "curve": curve_rows, "epochs": epochs,
            "train_seconds": round(time.time() - t0, 1),
            "lambda_drift": {c: dict(zip(map(str, _data.HORIZONS), np.round(lam[i], 4).tolist()))
                             for i, c in enumerate(_data.COMMODITIES)} if model.drift else None}


def fit_predict(fd, cfg: dict, seed: int, log=print, curve: bool = False) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    model = build(cfg, fd)
    log(f"  arifnet: {_trainer.n_params(model):,} parameter, L={fd.L}, encoder={model.encoder}, "
        f"hurdle={model.hurdle}, kuantil={len(model.taus)}, drift={model.drift}, modality+={model.ext}")
    info = fit(model, fd, cfg, seed, log=log, curve=curve)
    return {"r_hat": predict(model, fd, fd.val, info["sigma"])}, {**info, "n_params": _trainer.n_params(model)}
