"""
ARIF-Net — satu pintu menjalankan model apa pun pada satu peran data (F1, F2, atau final → test) dan mencatat hasilnya.

Konfigurasi: configs/models/<model>.json, kunci "family" ∈ persistence | arima | tabular | neural | tcn_xgb |
arifnet | arifnet_gate (model usulan, 04_arif_net/).
Override kunci bertitik (mis. {"model.d": 64, "train.lr": 1e-4}) dipakai pencarian.
Catatan run: results/runs/<run_id>/{result.json, predictions.csv}.
"""
from __future__ import annotations

import copy
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import trainer
from .data import CONFIGS, RESULTS, FoldData, load_fold, shock_thresholds, COMMODITIES, HORIZONS
from .metrics import overall, table

RUNS = RESULTS / "runs"
DEFAULT_L = 60


def load_config(model: str, overrides: dict | None = None, base: dict | None = None) -> dict:
    cfg = copy.deepcopy(base) if base is not None else json.loads(
        (CONFIGS / "models" / f"{model}.json").read_text(encoding="utf-8"))
    for k, v in (overrides or {}).items():
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = v
    return cfg


def seq_len(cfg: dict) -> int:
    if cfg["family"] == "tcn_xgb":
        return int(cfg["stage1"]["train"].get("L", 30))
    if cfg["family"] == "arifnet_gate":                         # expert ARIF-Net memakai L konfigurasi bekunya
        return int(json.loads((RESULTS / "frozen" / "arifnet.json").read_text(encoding="utf-8"))
                   ["config"]["train"]["L"])
    return int(cfg.get("train", {}).get("L", DEFAULT_L))


def env_info() -> dict:
    import sklearn, statsmodels, torch, xgboost, lightgbm
    return {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "host": platform.node()}


def predict_with(model: str, fd: FoldData, cfg: dict, seed: int, log=print, curve: bool = False) -> tuple[dict, dict]:
    fam = cfg["family"]
    if fam == "persistence":
        return {"r_hat": np.zeros(len(fd.val))}, {}
    if fam == "arima":
        from .models import arima
        out = arima.fit_predict(fd, cfg, seed, log=log)
        return out, {"arima": out.pop("info")}
    if fam == "tabular":
        from .models import tabular
        return tabular.fit_predict(fd, cfg, seed, log=log), {}
    if fam == "tcn_xgb":
        from .models import twostage
        return twostage.fit_predict(fd, cfg, seed, log=log)
    if fam == "neural":
        import torch
        from .models.neural import build_neural
        torch.manual_seed(seed)
        net = build_neural(cfg["model"].get("name", model), cfg, fd)
        log(f"  {model}: {trainer.n_params(net):,} parameter, L={fd.L}")
        info = trainer.fit(net, fd, cfg, seed, log=log, curve=curve)
        return {"r_hat": trainer.predict(net, fd, fd.val, info["sigma"])}, {**info, "n_params": trainer.n_params(net)}
    if fam == "arifnet":
        from importlib import import_module
        return import_module("04_arif_net.train").fit_predict(fd, cfg, seed, log=log, curve=curve)
    if fam == "arifnet_gate":
        from importlib import import_module
        return import_module("04_arif_net.gate").fit_predict(fd, cfg, seed, log=log)
    raise KeyError(f"family tidak dikenal: {fam}")


def insample_and_shock(fd: FoldData) -> tuple[dict, dict]:
    """Penyebut MASE (MAE persistence data latih) dan ambang shock Q95 per (c, h), dari data LATIH."""
    T = fd.train
    thr = shock_thresholds(T)
    ins, sh = {}, {}
    for ci, c in enumerate(COMMODITIES):
        for hi, h in enumerate(HORIZONS):
            v = T.y[(T.c == ci) & (T.hi == hi)]
            ins[(c, h)] = float(np.mean(np.abs(v))) if len(v) else None
            sh[(c, h)] = thr[ci, hi]
    return ins, sh


def evaluate(fd: FoldData, r_hat: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not np.all(np.isfinite(r_hat)):
        raise RuntimeError("prediksi mengandung NaN/inf")
    pred = fd.frames_val[["comcat_id", "h", "origin_date", "label_date", "price_origin", "price_label", "r"]].copy()
    pred["r_hat"] = r_hat
    ins, sh = insample_and_shock(fd)
    return pred, table(pred, ins, sh), overall(pred)


def run(model: str, role: str, seed: int = 0, cfg: dict | None = None, tag: str = "", log=print,
        curve: bool = False, save: bool = True, out_dir=None) -> dict:
    cfg = cfg or load_config(model)
    fd = load_fold(role, seq_len(cfg))
    t0 = time.time()
    out, info = predict_with(model, fd, cfg, seed, log=log, curve=curve)
    pred, tab, ov = evaluate(fd, out["r_hat"])
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{model}{('_' + tag) if tag else ''}_{role}_s{seed}"
    summary = {"run_id": run_id, "model": model, "role": role, "seed": seed, "tag": tag,
               "seconds": round(time.time() - t0, 1), "train_rows": len(fd.train), "eval_rows": len(fd.val),
               "eval_period": [str(fd.frames_val["origin_date"].min().date()),
                               str(fd.frames_val["origin_date"].max().date())],
               **ov, "curve": info.get("curve", []), "config": cfg,
               "info": {k: v for k, v in info.items() if k not in ("sigma", "curve", "history")}}
    if save:
        d = (out_dir or RUNS) / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        pred.to_csv(d / "predictions.csv", index=False)
        tab.to_csv(d / "metrics_per_ch.csv", index=False)
    log(f"  {model} {role} s{seed}: relMAE {ov['relmae']:.4f}  MAE {ov['mae']:.5f}  RMSE {ov['rmse']:.5f}"
        f"  DA {ov['dir_acc']:.3f}  ({summary['seconds']}s)")
    summary["predictions"] = pred
    return summary
