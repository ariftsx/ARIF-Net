"""
ARIF-Net — CLI training baseline & model penelitian terdahulu (uji komparasi sebelum desain ARIF-Net).

Alur (PROTOKOL_EKSPERIMEN §2, disesuaikan codebase baru: validasi F1–F2, F3 = peran final):
  1. search    konfigurasi default paper + grid kecil, F1–F2, seed 0 → bekukan results/frozen/<model>.json
  2. validate  konfigurasi beku × F1–F2 × 5 seed (neural) → results/validation/<model>.csv
  3. test      latih ulang pada peran final → prediksi test SEKALI (kunci results/test/OPENED.json)
  run          satu model × satu fold (uji coba / debugging), tidak membekukan apa pun

ARIF-Net (docs/ARIFNET_SPEC.md, docs/PROTOKOL_UJI_PROSPEKTIF.md):
  search --model arifnet       ≤ 40 kandidat F1–F2 + cek ulang top-3 × 5 seed → bekukan
  validate --model arifnet     A0 beku × F1–F2 × 5 seed
  ablate --model arifnet       varian C-1..C-5, E-1, M-1..M-4, X-1 dari A0 beku → validasi 5 seed
  register --model all         registrasi uji prospektif (semua model beku), SEBELUM data sesudah 2026-09-30 dilihat
  prospective --model all      pembukaan uji prospektif sekali (sesudah data sampai cutoff dikumpulkan)

Contoh:
  python -m 03_training.training search --model all
  python -m 03_training.training validate --model all
  python -m 03_training.training test --model all [--dry-run]
  python -m 03_training.training run --model gru --fold F1 --seed 0 --set train.epochs=5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback

from .data import VALIDATION_FOLDS

MODELS = ("persistence", "arima", "ridge", "svr", "rf", "xgb", "lgbm",
          "lstm", "gru", "bilstm", "dialstm", "tft", "tcn_xgb")      # baseline: makna `--model all` untuk search/validate
ARIF = ("arifnet",)


def _known(m: str) -> bool:
    from .data import CONFIGS, RESULTS
    return m in MODELS + ARIF or (RESULTS / "frozen" / f"{m}.json").exists() or (CONFIGS / "models" / f"{m}.json").exists()


def _all_frozen() -> list[str]:
    from .data import RESULTS
    return sorted(f.stem for f in (RESULTS / "frozen").glob("*.json"))


def _parse_set(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ARIF-Net training baseline & model terdahulu.")
    ap.add_argument("cmd", choices=["run", "search", "validate", "test", "ablate", "register", "prospective"])
    ap.add_argument("--model", required=True, help=f"{', '.join(MODELS)} atau 'all' (bisa dipisah koma)")
    ap.add_argument("--fold", default="F1", help=f"untuk `run`: {', '.join(VALIDATION_FOLDS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="override config untuk `run`, mis. train.epochs=5 model.d=64")
    ap.add_argument("--dry-run", action="store_true", help="`test`: gladi resik pada F2, data test tidak disentuh")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):                    # konsol Windows (cp1252) tidak bisa cetak ▶ × σ
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.info

    if args.model == "all":
        models = _all_frozen() if args.cmd in ("register", "prospective") else list(MODELS)
    else:
        models = args.model.split(",")
    unknown = [m for m in models if not _known(m)]
    if unknown:
        log(f"model tidak dikenal: {unknown}")
        return 2

    if args.cmd == "test":
        from .final_test import run_test
        return run_test(models, log=log, dry_run=args.dry_run)
    if args.cmd == "register":
        from .final_test import register
        return register(models, log=log)
    if args.cmd == "prospective":
        from .final_test import run_prospective
        return run_prospective(models, log=log)

    failed = []
    for m in models:
        try:
            if args.cmd == "run":
                from . import runs
                runs.run(m, args.fold, args.seed, cfg=runs.load_config(m, _parse_set(args.set)), tag="coba", log=log)
            elif args.cmd == "search":
                from .search import search
                search(m, log=log)
            elif args.cmd == "validate":
                from .final_test import validate
                validate(m, log=log)
            elif args.cmd == "ablate":
                from .final_test import ablate
                ablate(m, log=log)
        except Exception:                                      # noqa: BLE001 — dicatat, lalu exit code ≠ 0
            failed.append(m)
            logging.error("GAGAL %s:\n%s", m, traceback.format_exc())
    if failed:
        logging.error("Model gagal: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
