"""
ARIF-Net — pencarian konfigurasi pada fold validasi F1–F2 (PROTOKOL_EKSPERIMEN P4-DG-05; kebijakan "paper default
+ search").

configs/search/<model>.json → {"budget": n, "space": {"model.d": [...], "train.lr": [...]}}.
Kandidat 0 = konfigurasi default (= hyperparameter paper asli, configs/models/<model>.json), lalu sampel acak tanpa
pengulangan dari grid (seed sampling tetap 2026) sampai `budget`. Seed latih 0.
Skor seleksi = rata-rata relMAE F1 dan F2 (bobot sama). Model neural: jumlah epoch dipilih bersama hyperparameter
lain dari kurva checkpoint (trainer.CHECKPOINT_EPOCHS).
Hasil: results/search/<model>.csv, <model>_peringkat.csv; konfigurasi terbaik dibekukan ke results/frozen/<model>.json.
"""
from __future__ import annotations

import itertools
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import runs
from .data import CONFIGS, RESULTS, VALIDATION_FOLDS

SEARCH = RESULTS / "search"
FROZEN = RESULTS / "frozen"
CURVE_FAMILIES = ("neural", "arifnet")                     # epoch = hyperparameter dari kurva checkpoint


def candidates(model: str) -> list[dict]:
    p = CONFIGS / "search" / f"{model}.json"
    if not p.exists():
        return [{}]
    spec = json.loads(p.read_text(encoding="utf-8"))
    keys = list(spec["space"])
    grid = [dict(zip(keys, v)) for v in itertools.product(*(spec["space"][k] for k in keys))]
    random.Random(2026).shuffle(grid)
    base = runs.load_config(model)
    out = [{}]
    for g in grid:
        if len(out) >= spec["budget"]:
            break
        if runs.load_config(model, g) != base:                # lewati kandidat yang identik dengan default
            out.append(g)
    return out


def search(model: str, folds=VALIDATION_FOLDS, seed: int = 0, log=print) -> dict:
    cands = candidates(model)
    SEARCH.mkdir(parents=True, exist_ok=True)
    neural = runs.load_config(model)["family"] in CURVE_FAMILIES
    part = SEARCH / f"{model}.partial.csv"                    # simpanan per run → pencarian dapat dilanjutkan
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(int(r["cand"]), r["fold"]) for r in rows}
    log(f"▶ pencarian {model}: {len(cands)} kandidat × {len(folds)} fold (seed {seed})"
        + (" · kurva epoch" if neural else "") + (f" · melanjutkan {len(done)} run" if done else ""))
    for fold in folds:
        for i, ov in enumerate(cands):
            if (i, fold) in done:
                continue
            cfg = runs.load_config(model, ov)
            s = runs.run(model, fold, seed, cfg=cfg, tag=f"cari{i:02d}", log=lambda *_: None, curve=neural)
            pts = s["curve"] if neural else [{"epoch": None, "relmae": s["relmae"]}]
            for pt in pts:
                rows.append({"cand": i, "fold": fold, "epochs": pt["epoch"], "relmae": pt["relmae"],
                             "run_id": s["run_id"], "overrides": json.dumps(ov)})
            pd.DataFrame(rows).to_csv(part, index=False)
            b = min(pts, key=lambda x: x["relmae"])
            log(f"  {fold} kandidat {i:02d}: relMAE {b['relmae']:.4f}"
                + (f" (epoch {b['epoch']})" if neural else "") + f"  {ov or '(default paper)'}")
    df = pd.DataFrame(rows)
    df.to_csv(SEARCH / f"{model}.csv", index=False)
    part.unlink(missing_ok=True)
    key = ["cand", "overrides"] + (["epochs"] if neural else [])
    piv = df.pivot_table(index=key, columns="fold", values="relmae", dropna=False).reset_index()
    piv = piv.dropna(subset=list(folds))
    if neural:
        piv = piv[piv["epochs"] > 0]                          # epoch 0 = persistence, bukan konfigurasi model
    piv["skor"] = piv[list(folds)].mean(axis=1)
    piv = piv.sort_values("skor").reset_index(drop=True)
    piv.to_csv(SEARCH / f"{model}_peringkat.csv", index=False)
    best = piv.iloc[0]
    recheck = _spec(model).get("recheck_top", 0)
    if recheck:                                              # SPEC §4: top-k dicek ulang 5 seed, pilih rata-rata terbaik
        best = _recheck(model, piv.head(recheck), folds, neural, log)
    ov = json.loads(best["overrides"])
    if neural:
        ov["train.epochs"] = int(best["epochs"])
    cfg = runs.load_config(model, ov)
    FROZEN.mkdir(parents=True, exist_ok=True)
    frozen = {"model": model, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "selection": "rata-rata relMAE F1–F2, seed 0 (PROTOKOL P4-DG-05; epoch = hyperparameter v1.0.1)",
              "score": float(best["skor"]), "per_fold": {f: float(best[f]) for f in folds},
              "n_candidates": len(cands), "overrides": ov, "config": cfg}
    (FROZEN / f"{model}.json").write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"  BEKU {model}: skor {best['skor']:.4f} ({', '.join(f'{f} {best[f]:.4f}' for f in folds)}) · {ov or 'default'}")
    return frozen


def _spec(model: str) -> dict:
    p = CONFIGS / "search" / f"{model}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _recheck(model: str, top: pd.DataFrame, folds, neural: bool, log) -> pd.Series:
    from .data import SEEDS
    rows = []
    for _, r in top.iterrows():
        ov = json.loads(r["overrides"])
        if neural:
            ov["train.epochs"] = int(r["epochs"])
        cfg = runs.load_config(model, ov)
        sc = {f: np.mean([runs.run(model, f, s, cfg=cfg, tag=f"cek{int(r['cand']):02d}", log=lambda *_: None)["relmae"]
                          for s in SEEDS]) for f in folds}
        rows.append({**r.to_dict(), **sc, "skor": float(np.mean(list(sc.values())))})
        log(f"  cek ulang kandidat {int(r['cand']):02d} ({len(SEEDS)} seed): skor {rows[-1]['skor']:.4f}")
    df = pd.DataFrame(rows).sort_values("skor").reset_index(drop=True)
    df.to_csv(SEARCH / f"{model}_cek_ulang.csv", index=False)
    return df.iloc[0]


def frozen(model: str) -> dict:
    p = FROZEN / f"{model}.json"
    if not p.exists():
        raise FileNotFoundError(f"{model} belum dibekukan -- jalankan `python run.py train search --model {model}`")
    return json.loads(p.read_text(encoding="utf-8"))["config"]
