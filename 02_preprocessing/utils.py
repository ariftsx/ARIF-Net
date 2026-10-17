"""
ARIF-Net — Utilitas preprocessing bersama.

Shared functions untuk seluruh preprocessing pipeline: I/O, path, normalisasi.
"""
from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FINAL = ROOT / "data" / "final"
RESULTS = ROOT / "results"

WIB = timezone(timedelta(hours=7))

COMMODITIES = ("com_14", "com_11", "com_3")
COMMODITY_NAMES = {
    "com_14": "Cabai Merah Keriting",
    "com_11": "Bawang Merah Ukuran Sedang",
    "com_3": "Beras Kualitas Medium I",
}
HORIZONS = (1, 3, 7, 14, 30, 90, 180, 365)
FOLDS = ("F1", "F2", "F3")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_project_config() -> dict:
    return load_json(CONFIG / "project.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
