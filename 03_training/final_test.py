"""
ARIF-Net — VALIDASI LENGKAP dan UJI AKHIR SEKALI BUKA (PROTOKOL_EKSPERIMEN P4-DG-07, P4-DG-11).

validate : konfigurasi beku × F1–F2 × seed (neural & TCN-XGB: 5 seed, deterministik: 1) → results/validation/
test     : kunci results/test/OPENED.json ditulis SEBELUM data test dibaca; pemanggilan kedua sesudah selesai ditolak.
           Tiap model dilatih ulang pada peran final (origin & label < test_start) dengan konfigurasi beku, lalu
           memprediksi test sekali. Prediksi test model multi-seed = rata-rata seed (ensembel seed).
           --dry-run: gladi resik memakai F2 sebagai pengganti test → results/test_gladi/ (data test tidak disentuh).

ablate   : varian ablation ARIF-Net (configs/ablations/arifnet.json) = konfigurasi beku A0 + override → dibekukan
           sebagai results/frozen/arifnet__<varian>.json lalu divalidasi 5 seed (docs/ARIFNET_SPEC.md §3).
register : registrasi uji prospektif (docs/PROTOKOL_UJI_PROSPEKTIF.md §3 langkah 4): snapshot data latih
           prospektif + sha256 konfigurasi beku, kode, dan data → results/prospektif/REGISTERED.json.
prospective: pembukaan uji prospektif SEKALI; menolak bila konfigurasi/kode/snapshot berubah sejak registrasi.

Catatan status test (keputusan peneliti 2026-10-23): periode test 2024-07-01 → 2026-09-30 sama dengan eksperimen
codebase lama (dibuka 2026-10-16), sehingga hasilnya dilaporkan sebagai uji pada periode yang sudah pernah dilihat
peneliti; kunci tetap dipakai agar konfigurasi tidak diubah sesudah melihat hasil.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd

from . import runs
from .data import CONFIGS, DATA_FINAL, PROJECT, RESULTS, ROOT, SEEDS, VALIDATION_FOLDS
from .metrics import overall
from .search import frozen

VALIDATION = RESULTS / "validation"
TEST = RESULTS / "test"
DRY = RESULTS / "test_gladi"
PROSPEKTIF = RESULTS / "prospektif"
MULTI_SEED = ("neural", "tcn_xgb", "arifnet", "arifnet_gate")
CODE_DIRS = ("01_collection", "02_preprocessing", "03_training", "04_arif_net")


def seeds_for(cfg: dict) -> tuple:
    return SEEDS if cfg["family"] in MULTI_SEED else (0,)


def validate(model: str, log=print) -> pd.DataFrame:
    cfg = frozen(model)
    rows = []
    for fold in VALIDATION_FOLDS:
        for s in seeds_for(cfg):
            r = runs.run(model, fold, s, cfg=cfg, tag="valid", log=log)
            rows.append({"model": model, "fold": fold, "seed": s,
                         **{k: r[k] for k in ("relmae", "mae", "rmse", "dir_acc", "r_squared", "n")}})
    df = pd.DataFrame(rows)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    df.to_csv(VALIDATION / f"{model}.csv", index=False)
    agg = df.groupby("fold")["relmae"].agg(["mean", "std"])
    log(f"  VALIDASI {model}: " + ", ".join(f"{f} {agg.loc[f, 'mean']:.4f}±{agg.loc[f, 'std'] if pd.notna(agg.loc[f, 'std']) else 0:.4f}"
                                           for f in agg.index) + f" · skor {agg['mean'].mean():.4f}")
    return df


def ablate(base: str = "arifnet", log=print) -> list[str]:
    """Bekukan varian ablation dari konfigurasi beku A0 (hyperparameter & epoch identik), lalu validasi."""
    spec = json.loads((CONFIGS / "ablations" / f"{base}.json").read_text(encoding="utf-8"))
    src = json.loads((RESULTS / "frozen" / f"{base}.json").read_text(encoding="utf-8"))
    names = []
    for code, ov in spec["variants"].items():
        name = f"{base}__{code}"
        cfg = runs.load_config(base, ov, base=src["config"])
        (RESULTS / "frozen" / f"{name}.json").write_text(json.dumps(
            {"model": name, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "selection": f"ablation {code} dari {base} beku ({src['frozen_utc']})", "overrides": ov,
             "score": None, "per_fold": {}, "config": cfg}, indent=1, ensure_ascii=False), encoding="utf-8")
        names.append(name)
    for name in names:
        if (VALIDATION / f"{name}.csv").exists():
            log(f"  {name}: validasi sudah ada, dilewati")
            continue
        validate(name, log=log)
    return names


def run_test(models: list[str], log=print, dry_run: bool = False) -> int:
    out_dir = DRY if dry_run else TEST
    role = VALIDATION_FOLDS[-1] if dry_run else "final"
    return _open(models, role, out_dir, log, dry_run, note="Periode test sama dengan eksperimen codebase lama.")


# ───────────────────────────────────────────── uji prospektif
def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_hashes() -> dict:
    return {f.relative_to(ROOT).as_posix(): _sha(f) for d in CODE_DIRS for f in sorted((ROOT / d).rglob("*.py"))
            if "__pycache__" not in f.parts}


def _git() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def register(models: list[str], log=print) -> int:
    reg = PROSPEKTIF / "REGISTERED.json"
    if reg.exists():
        raise PermissionError(f"Uji prospektif sudah teregistrasi ({reg}). Registrasi hanya sekali.")
    cfgs = {m: frozen(m) for m in models}
    src = DATA_FINAL / "dataset_prospektif_train.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} tidak ada -- jalankan `python run.py preprocess`")
    tr = pd.read_csv(src, usecols=["origin_date", "label_date"])
    p = PROJECT["prospective"]
    if tr["label_date"].max() > p["train_end"]:
        raise RuntimeError("snapshot latih memuat label sesudah train_end")
    PROSPEKTIF.mkdir(parents=True, exist_ok=True)
    snap = PROSPEKTIF / "train_snapshot.csv"
    shutil.copyfile(src, snap)
    body = {"registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol": p["protocol"], "train_end": p["train_end"], "start": p["start"], "cutoff": p["cutoff"],
            "models": models, "frozen": cfgs, "train_snapshot_sha256": _sha(snap), "code_sha256": _code_hashes(),
            "git": _git(), "env": runs.env_info(),
            "criteria": "docs/PROTOKOL_UJI_PROSPEKTIF.md §5 (K-1..K-5), didaftarkan sebelum hasil"}
    reg.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"▶ REGISTRASI uji prospektif: {len(models)} model, cutoff {p['cutoff']}, snapshot {len(tr)} baris"
        + (" · PERINGATAN: ada perubahan git belum di-commit" if body["git"]["dirty"] else ""))
    return 0


def run_prospective(models: list[str], log=print) -> int:
    reg = PROSPEKTIF / "REGISTERED.json"
    if not reg.exists():
        raise PermissionError("Belum teregistrasi -- jalankan `python run.py train register --model all` dulu.")
    R = json.loads(reg.read_text(encoding="utf-8"))
    if set(models) != set(R["models"]):
        raise PermissionError(f"Daftar model berbeda dari registrasi: {sorted(set(models) ^ set(R['models']))}")
    cfgs = {m: frozen(m) for m in models}
    if json.dumps(cfgs, sort_keys=True) != json.dumps(R["frozen"], sort_keys=True):
        raise PermissionError("Konfigurasi beku berubah sejak registrasi -- dilarang.")
    now = _code_hashes()
    changed = sorted(k for k in set(now) | set(R["code_sha256"]) if now.get(k) != R["code_sha256"].get(k))
    if changed:
        raise PermissionError(f"Kode berubah sejak registrasi: {changed}")
    if _sha(PROSPEKTIF / "train_snapshot.csv") != R["train_snapshot_sha256"]:
        raise PermissionError("Snapshot data latih berubah sejak registrasi -- dilarang.")
    if not (DATA_FINAL / "dataset_prospektif_test.csv").exists():
        raise FileNotFoundError("dataset_prospektif_test.csv belum ada: kumpulkan data sampai cutoff lalu preprocess.")
    return _open(models, "prospektif", PROSPEKTIF, log, False,
                 note=f"Uji prospektif sekali buka, registrasi {R['registered_utc']}.")


def _open(models: list[str], role: str, out_dir, log, dry_run: bool, note: str) -> int:
    cfgs = {m: frozen(m) for m in models}                     # semua model harus sudah beku
    lock = out_dir / "OPENED.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if lock.exists() and not dry_run:
        prev = json.loads(lock.read_text(encoding="utf-8"))
        if "finished_utc" in prev:
            raise PermissionError(f"Peran {role} sudah dibuka dan selesai ({prev['opened_utc']}). Hanya sekali.")
        if json.dumps(prev["frozen"], sort_keys=True) != json.dumps(cfgs, sort_keys=True):
            raise PermissionError("Pembukaan test terputus dan konfigurasi beku BERUBAH sejak itu -- dilarang.")
        prev.setdefault("dilanjutkan", []).append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
        lock.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
        log("▶ UJI AKHIR -- melanjutkan pembukaan yang terputus (konfigurasi beku identik)")
    else:
        lock.write_text(json.dumps({"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "dry_run": dry_run, "role": role, "models": models, "frozen": cfgs,
                                    "env": runs.env_info(), "note": note},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"▶ {'GLADI RESIK (F2)' if dry_run else 'PEMBUKAAN SEKALI (' + role + ')'}: {', '.join(models)}")
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    rows, failed = [], []
    for m in models:
        f = pred_dir / f"{m}.csv"
        if f.exists():
            log(f"  {m}: sudah ada, dilewati")
            continue
        try:
            preds = []
            for s in seeds_for(cfgs[m]):
                r = runs.run(m, role, s, cfg=cfgs[m], tag="test", log=log, out_dir=out_dir / "runs")
                preds.append(r["predictions"].rename(columns={"r_hat": f"r_hat_s{s}"}))
            base = preds[0].drop(columns=[c for c in preds[0].columns if c.startswith("r_hat_s")])
            for p in preds:
                base = base.join(p[[c for c in p.columns if c.startswith("r_hat_s")]])
            base["r_hat"] = base[[c for c in base.columns if c.startswith("r_hat_s")]].mean(axis=1)
            base.to_csv(f, index=False)
            rows.append({"model": m, **overall(base)})
        except Exception as exc:                              # noqa: BLE001 — dicatat lalu dilaporkan sebagai gagal
            failed.append(m)
            log(f"  GAGAL {m}: {exc!r}")
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"summary_{datetime.now():%Y%m%d-%H%M%S}.csv", index=False)
    if not failed and not dry_run:
        prev = json.loads(lock.read_text(encoding="utf-8"))
        prev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lock.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0
