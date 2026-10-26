r"""
ARIF-Net -- SATU PINTU MASUK seluruh pipeline.

    python run.py status                  ringkasan tahap mana yang sudah/belum
    python run.py collect   [--step ...]  tahap 1: collection semua data
    python run.py preprocess              tahap 2: preprocessing & build dataset
    python run.py train search   --model all    tahap 3a: cari konfigurasi (default paper + grid), F1-F2, bekukan
    python run.py train validate --model all    tahap 3b: konfigurasi beku x F1-F2 x 5 seed
    python run.py train test     --model all    tahap 3c: latih final -> test SEKALI (--dry-run = gladi di F2)
    python run.py train run --model gru --fold F1   uji coba satu model
    python run.py analyze [--source test]       tahap 4: gap analysis -> rekomendasi ARIF-Net
    python run.py full                    jalankan semua tahap secara berurutan

Jalankan dari root repo:
    cd "d:\.Capstone dan TA\ARIF-Net"
    python run.py <perintah>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, "-u", "-X", "utf8"]


# ──────────────────────────────────── helpers
def step(name: str, cmd: list[str], cwd: Path | None = None) -> dict:
    """Jalankan sub-perintah, cetak hasil."""
    cwd = cwd or ROOT
    print(f"\n> {name}\n  {' '.join(cmd[-3:])}", flush=True)
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    raw = (p.stdout + p.stderr).strip()
    # Sanitize non-ASCII for Windows console
    safe = raw.encode("ascii", "replace").decode("ascii")
    tail = [ln for ln in safe.splitlines() if ln.strip()][-5:]
    ok = p.returncode == 0
    print(f"  {'LULUS' if ok else 'GAGAL'} ({time.time() - t:.0f}s)")
    for ln in tail:
        print(f"    {ln}")
    return {"step": name, "ok": ok, "returncode": p.returncode,
            "seconds": round(time.time() - t, 1), "tail": tail}


# ──────────────────────────────────── STATUS
STAGES = [
    ("1a", "Collect PIHPS (harga komoditas)",   "01_collection.collect_pihps",  "data/raw/pihps/"),
    ("1b", "Collect News (berita + isi)",        "01_collection.collect_news",   "data/raw/news/"),
    ("1c", "Collect Climate (iklim)",            "01_collection.collect_climate", "data/raw/climate/"),
    ("1d", "Collect BBM (bahan bakar)",          "01_collection.collect_bbm",    "data/raw/bbm/"),
    ("1e", "Collect Calendar (kalender)",        "01_collection.collect_calendar","data/raw/calendar/"),
    ("2a", "Preprocess PIHPS",                   "02_preprocessing.preprocess_pihps", "data/processed/pihps/"),
    ("2b", "Preprocess News",                    "02_preprocessing.preprocess_news",  "data/processed/news/"),
    ("2c", "Preprocess Climate",                 "02_preprocessing.preprocess_climate","data/processed/climate/"),
    ("2d", "Build Target r(t,h)",                "02_preprocessing.build_target",  "data/processed/target/"),
    ("2e", "Build Dataset Final",                "02_preprocessing.build_dataset", "data/final/"),
    ("3",  "Train Baseline Models",              "03_training.training",          "results/runs/"),
    ("4",  "Gap Analysis",                       "03_training.analyze_gaps",      "results/gap_analysis/"),
    ("5",  "ARIF-Net (setelah analisis)",        "04_arif_net",                   "04_arif_net/"),
]


def cmd_status(args) -> int:
    print("ARIF-Net -- Pipeline Status\n")
    print(f"{'Tahap':<7}{'Nama':<40}{'Output':<30}{'Status'}")
    print("-" * 100)
    for key, name, module, output in STAGES:
        out_path = ROOT / output
        if out_path.exists() and any(out_path.iterdir() if out_path.is_dir() else [out_path]):
            n_files = sum(1 for _ in out_path.rglob("*") if _.is_file()) if out_path.is_dir() else 1
            status = f"[OK] {n_files} file"
        else:
            status = "-- belum"
        print(f"{key:<7}{name:<40}{output:<30}{status}")
    return 0


# ──────────────────────────────────── COLLECT
def cmd_collect(args) -> int:
    results = []
    steps_map = {
        "pihps":    ("PIHPS", PY + ["-m", "01_collection.collect_pihps", "full"]),
        "news_idx": ("News Index", PY + ["-m", "01_collection.collect_news", "index",
                                         "--max-minutes", str(args.max_minutes or 120)]),
        "news_art": ("News Articles", PY + ["-m", "01_collection.collect_news", "articles",
                                            "--max-minutes", str(args.max_minutes or 120)]),
        "news_norm":("News Normalize", PY + ["-m", "01_collection.collect_news", "normalize"]),
        "climate":  ("Climate", PY + ["-m", "01_collection.collect_climate"]),
        "bbm":      ("BBM", PY + ["-m", "01_collection.collect_bbm"]),
        "calendar": ("Calendar", PY + ["-m", "01_collection.collect_calendar"]),
    }

    if args.step:
        todo = {s: steps_map[s] for s in args.step if s in steps_map}
    else:
        todo = steps_map

    for key, (name, cmd) in todo.items():
        results.append(step(f"Collection: {name}", cmd))

    ok = all(r["ok"] for r in results)
    print(f"\nCOLLECTION: {'SELESAI [OK]' if ok else 'ADA YANG GAGAL [X]'}")
    return 0 if ok else 1


# ──────────────────────────────────── PREPROCESS
def cmd_preprocess(args) -> int:
    results = [
        step("Preprocess PIHPS", PY + ["-m", "02_preprocessing.preprocess_pihps"]),
        step("Preprocess News", PY + ["-m", "02_preprocessing.preprocess_news"]),
        step("Preprocess Climate", PY + ["-m", "02_preprocessing.preprocess_climate"]),
        step("Build Target r(t,h)", PY + ["-m", "02_preprocessing.build_target"]),
        step("Build Dataset Final", PY + ["-m", "02_preprocessing.build_dataset"]),
    ]
    ok = all(r["ok"] for r in results)
    print(f"\nPREPROCESSING: {'SELESAI [OK]' if ok else 'ADA YANG GAGAL [X]'}")
    return 0 if ok else 1


# ──────────────────────────────────── TRAIN
def cmd_train(args) -> int:
    cmd = PY + ["-m", "03_training.training", args.action, "--model", args.model,
                "--fold", args.fold, "--seed", str(args.seed)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.set:
        cmd += ["--set", *args.set]
    print(f"\n> Train {args.action} {args.model}")
    t = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode          # log tampil langsung (proses panjang)
    print(f"  {'LULUS' if rc == 0 else 'GAGAL'} ({time.time() - t:.0f}s)")
    return rc


# ──────────────────────────────────── ANALYZE (GAP ANALYSIS)
def cmd_analyze(args) -> int:
    r = step("Gap Analysis", PY + ["-m", "03_training.analyze_gaps", "--source", args.source])
    if r["ok"]:
        report = ROOT / "results" / "gap_analysis" / "gap_analysis_report.md"
        if report.exists():
            print(f"\nLaporan: {report}")
    return 0 if r["ok"] else 1


# ──────────────────────────────────── FULL PIPELINE
def cmd_full(args) -> int:
    """Jalankan seluruh pipeline dari awal."""
    print("=" * 60)
    print("ARIF-Net -- Full Pipeline Execution")
    print("=" * 60)

    t0 = time.time()

    # Tahap 1: Collection
    print("\n" + "=" * 60)
    print("TAHAP 1: DATA COLLECTION")
    print("=" * 60)

    steps_1 = [
        step("Collect BBM", PY + ["-m", "01_collection.collect_bbm"]),
        step("Collect Calendar", PY + ["-m", "01_collection.collect_calendar"]),
        step("Collect Climate", PY + ["-m", "01_collection.collect_climate"]),
    ]

    # PIHPS & News memerlukan waktu lama, jalankan terpisah
    print("\n[!] PIHPS dan News collection memerlukan waktu lama.")
    print("    Jalankan manual jika belum ada data:")
    print("    python run.py collect --step pihps")
    print("    python run.py collect --step news_idx news_art news_norm")

    # Tahap 2: Preprocessing
    print("\n" + "=" * 60)
    print("TAHAP 2: PREPROCESSING")
    print("=" * 60)

    steps_2 = [
        step("Preprocess PIHPS", PY + ["-m", "02_preprocessing.preprocess_pihps"]),
        step("Preprocess News", PY + ["-m", "02_preprocessing.preprocess_news"]),
        step("Preprocess Climate", PY + ["-m", "02_preprocessing.preprocess_climate"]),
        step("Build Target", PY + ["-m", "02_preprocessing.build_target"]),
        step("Build Dataset", PY + ["-m", "02_preprocessing.build_dataset"]),
    ]

    # Tahap 3: Training
    print("\n" + "=" * 60)
    print("TAHAP 3: TRAINING BASELINE MODELS")
    print("=" * 60)

    # Pencarian + validasi semua model (lama; test sengaja TIDAK dijalankan otomatis -- sekali buka, manual)
    steps_3 = [step(f"Train {a}", PY + ["-m", "03_training.training", a, "--model", "all"])
               for a in ("search", "validate")]

    # Gap Analysis
    print("\n" + "=" * 60)
    print("TAHAP 4: GAP ANALYSIS")
    print("=" * 60)
    steps_4 = [step("Gap Analysis", PY + ["-m", "03_training.analyze_gaps"])]

    all_steps = steps_1 + steps_2 + steps_3 + steps_4
    n_ok = sum(1 for s in all_steps if s["ok"])
    n_fail = sum(1 for s in all_steps if not s["ok"])
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"PIPELINE SELESAI: {n_ok} lulus, {n_fail} gagal ({elapsed/60:.1f} menit)")
    print(f"{'=' * 60}")
    return 0 if n_fail == 0 else 1


# ──────────────────────────────────── MAIN
def main() -> int:
    ap = argparse.ArgumentParser(description="ARIF-Net pipeline -- satu pintu masuk.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Lihat status pipeline").set_defaults(f=cmd_status)

    p = sub.add_parser("collect", help="Tahap 1: Collection data")
    p.add_argument("--step", nargs="+",
                   choices=["pihps", "news_idx", "news_art", "news_norm",
                            "climate", "bbm", "calendar"],
                   help="Step tertentu (default: semua)")
    p.add_argument("--max-minutes", type=float, default=None)
    p.set_defaults(f=cmd_collect)

    sub.add_parser("preprocess", help="Tahap 2: Preprocessing").set_defaults(f=cmd_preprocess)

    p = sub.add_parser("train", help="Tahap 3: search / validate / test / run")
    p.add_argument("action", choices=["search", "validate", "test", "run", "ablate", "register", "prospective"])
    p.add_argument("--model", required=True, help="Nama model, dipisah koma, atau 'all'")
    p.add_argument("--fold", default="F1", choices=["F1", "F2"], help="untuk `run`")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", help="override config untuk `run`, mis. train.epochs=5")
    p.add_argument("--dry-run", action="store_true", help="`test`: gladi resik di F2")
    p.set_defaults(f=cmd_train)

    p = sub.add_parser("analyze", help="Gap analysis -> rekomendasi ARIF-Net")
    p.add_argument("--source", choices=["validation", "test"], default="validation")
    p.set_defaults(f=cmd_analyze)
    sub.add_parser("full", help="Jalankan seluruh pipeline").set_defaults(f=cmd_full)

    args = ap.parse_args()
    return args.f(args)


if __name__ == "__main__":
    raise SystemExit(main())
