"""
ARIF-Net — training loop model NEURAL, dipakai sama oleh semua model neural (perlakuan identik).

1. σ_{c,h} dan ambang shock dihitung dari data LATIH saja; target dilatih dalam satuan r/σ_{c,h}.
2. Loss = MSE (default paper Manogna/Gu/Zhao) | Huber | MAE (median = kuantil 0,5, TFT/Amalia); optimizer Adam.
3. Jumlah epoch = hyperparameter (PROTOKOL v1.0.1): saat pencarian (curve=True) model dilatih sampai
   max(CHECKPOINT_EPOCHS) dan relMAE data evaluasi fold dicatat di tiap checkpoint, lalu epoch dipilih bersama
   hyperparameter lain dari rata-rata F1–F2. Alasan: sub-validasi kronologis di dalam data latih hampir hanya
   berisi horizon pendek (label horizon panjang terpotong purging), sehingga early stopping di dalamnya bias.
   Data test tidak pernah dipakai.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .data import FoldData, Tensors, resolve_threads, shock_thresholds, target_scale
from .metrics import relmae

CHECKPOINT_EPOCHS = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50)


def to_batch(fd: FoldData, T: Tensors, idx: np.ndarray, sig: np.ndarray, needs) -> dict:
    B = {"c": torch.as_tensor(T.c[idx], dtype=torch.long), "hi": torch.as_tensor(T.hi[idx], dtype=torch.long),
         "y": torch.as_tensor(T.y[idx] / sig[T.c[idx], T.hi[idx]], dtype=torch.float32)}
    for k in needs:
        if k == "seq":
            B[k] = torch.as_tensor(fd.seq_price[T.oi[idx]])
        elif k == "seq_clim":
            B[k] = torch.as_tensor(fd.seq_clim[T.oi[idx]])
        elif k.startswith("x_"):
            B[k] = torch.as_tensor(T.x[k[2:]][idx])
    return B


def _loss(name: str, delta: float):
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")
    if name == "huber":
        return nn.HuberLoss(delta=delta, reduction="none")
    raise KeyError(f"loss tidak dikenal: {name}")


def fit(model: nn.Module, fd: FoldData, cfg: dict, seed: int, log=print, curve: bool = False) -> dict:
    tc = cfg["train"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(resolve_threads(tc.get("threads", "auto")))
    T = fd.train
    sig = target_scale(T)
    thr = shock_thresholds(T)
    sw = float(tc.get("shock_weight", 1.0))
    crit = _loss(tc.get("loss", "mse"), float(tc.get("huber_delta", 1.0)))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    needs = model.needs
    batch = int(tc["batch"])
    epochs = max(CHECKPOINT_EPOCHS) if curve else int(tc["epochs"])
    curve_rows, hist = [], []
    t0 = time.time()

    def checkpoint(ep):
        if curve and ep in CHECKPOINT_EPOCHS:
            curve_rows.append({"epoch": ep, "relmae": relmae(fd.val.y, predict(model, fd, fd.val, sig))})

    checkpoint(0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(T))
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            B = to_batch(fd, T, idx, sig, needs)
            w = torch.ones(len(idx))
            if sw != 1.0:
                w = torch.as_tensor(np.where(np.abs(T.y[idx]) >= thr[T.c[idx], T.hi[idx]], sw, 1.0),
                                    dtype=torch.float32)
            loss = (crit(model(B), B["y"]) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            if tc.get("clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
            losses.append(float(loss.detach()))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)})
        checkpoint(ep)
        if ep in CHECKPOINT_EPOCHS or ep == epochs:
            log(f"    epoch {ep:3d}  loss latih {hist[-1]['train_loss']:.4f}"
                + (f"  relMAE evaluasi {curve_rows[-1]['relmae']:.4f}" if curve_rows else ""))
    return {"sigma": sig, "history": hist, "curve": curve_rows, "epochs": epochs,
            "train_seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def predict(model: nn.Module, fd: FoldData, T: Tensors, sig: np.ndarray, bs: int = 2048) -> np.ndarray:
    """Prediksi dalam satuan r."""
    model.eval()
    out = []
    for i in range(0, len(T), bs):
        idx = np.arange(i, min(i + bs, len(T)))
        r = model(to_batch(fd, T, idx, sig, model.needs)).numpy()
        out.append(r * sig[T.c[idx], T.hi[idx]])
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def embed(model: nn.Module, fd: FoldData, T: Tensors, sig: np.ndarray, bs: int = 2048) -> np.ndarray:
    """Representasi sebelum kepala keluaran (tahap 1 TCN-XGBoost)."""
    model.eval()
    return np.concatenate([model.embedding(to_batch(fd, T, np.arange(i, min(i + bs, len(T))), sig, model.needs)).numpy()
                           for i in range(0, len(T), bs)])


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
