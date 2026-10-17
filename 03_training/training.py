"""
ARIF-Net -- Training pipeline & evaluasi model.

Melatih dan mengevaluasi semua model baseline dan penelitian terdahulu
dengan aturan yang sama: data split, preprocessing, dan metrik identik.

Contoh:
  python -m 03_training.training --model ridge --fold F1
  python -m 03_training.training --model lstm --fold F1 --seed 42
  python -m 03_training.training --model all --fold all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from .metrics import basic, table, overall, mape_price

CONFIGS = ROOT / "configs"
DATA_FINAL = ROOT / "data" / "final"
RESULTS = ROOT / "results" / "runs"
FOLDS = ("F1", "F2", "F3")


def load_model_config(name: str) -> dict:
    path = CONFIGS / "models" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_data(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train dan val data untuk fold tertentu."""
    train = pd.read_csv(DATA_FINAL / f"dataset_{fold}_train.csv", parse_dates=["origin_date"])
    val = pd.read_csv(DATA_FINAL / f"dataset_{fold}_val.csv", parse_dates=["origin_date"])
    return train, val


def create_model(name: str, cfg: dict, train: pd.DataFrame, seed: int = 0):
    """Factory: buat model berdasarkan nama dan config."""
    model_cfg = cfg.get("model", cfg)
    mtype = model_cfg.get("type", name)

    if mtype == "persistence":
        from .models.persistence import PersistenceModel
        return PersistenceModel()

    if mtype == "arima":
        from .models.arima import ARIMAModel
        return ARIMAModel(order=model_cfg.get("order", (5, 1, 1)))

    if mtype == "tabular":
        from .models.tabular import TabularModel
        return TabularModel(
            kind=model_cfg["kind"],
            features=model_cfg.get("features", "multimodal"),
            params=model_cfg.get("params", {}),
            seed=seed,
        )

    if mtype == "neural":
        # Neural models require PyTorch training loop
        return _create_neural(name, cfg, train, seed)

    if mtype == "twostage":
        from .models.twostage import TwoStageModel
        return TwoStageModel(
            tcn_cfg=model_cfg.get("stage1", {}),
            xgb_params=model_cfg.get("stage2", {}).get("params", {}),
            seed=seed,
        )

    raise KeyError(f"Unknown model type: {mtype}")


def _create_neural(name: str, cfg: dict, train: pd.DataFrame, seed: int):
    """Placeholder: neural models memerlukan training loop khusus (lihat train_neural)."""
    from .models.neural import build_neural
    model_cfg = cfg.get("model", cfg)
    model_name = model_cfg.get("name", name)

    # Hitung dimensi fitur
    meta_cols = {"comcat_id", "commodity", "origin_date", "h",
                 "target_date_nominal", "label_date", "label_gap_days",
                 "price_origin", "price_label", "r",
                 "origin_holiday_carry", "origin_spike_reversal"}
    n_tabular = len([c for c in train.columns if c not in meta_cols])
    n_climate = len([c for c in train.columns if c.startswith("clim_")])
    n_calendar = len([c for c in train.columns if c.startswith("cal_")])

    return build_neural(model_name, model_cfg, n_tabular, n_climate, n_calendar)


def train_tabular(model, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Latih model tabular (sklearn-compatible) dan evaluasi."""
    t0 = time.time()
    y_train = train["r"].values
    model.fit(train, y_train)
    r_hat = model.predict(val)
    elapsed = time.time() - t0

    val_result = val.copy()
    val_result["r_hat"] = r_hat
    metrics_tab = table(val_result)
    metrics_overall = overall(metrics_tab)

    return {
        "model": str(model),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_seconds": round(elapsed, 1),
        "metrics_overall": metrics_overall,
        "metrics_per_ch": metrics_tab.to_dict("records"),
        "predictions": val_result[["comcat_id", "h", "origin_date", "r", "r_hat"]].to_dict("records"),
    }


def run_one(model_name: str, fold: str, seed: int = 0) -> dict:
    """Latih satu model pada satu fold."""
    logging.info("=" * 60)
    logging.info("Model: %s | Fold: %s | Seed: %d", model_name, fold, seed)
    logging.info("=" * 60)

    cfg = load_model_config(model_name)
    train, val = load_fold_data(fold)
    logging.info("Data: train=%d val=%d", len(train), len(val))

    model = create_model(model_name, cfg, train, seed)
    result = train_tabular(model, train, val)
    result.update({
        "model_name": model_name, "fold": fold, "seed": seed,
        "config": cfg,
    })

    # Simpan hasil
    run_id = f"{model_name}_{fold}_s{seed}_{int(time.time())}"
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simpan metrik (tanpa predictions yang besar)
    result_save = {k: v for k, v in result.items() if k != "predictions"}
    (out_dir / "result.json").write_text(
        json.dumps(result_save, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Simpan predictions
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    logging.info("Overall: relMAE=%.4f MAE=%.6f RMSE=%.6f DA=%.3f",
                 result["metrics_overall"]["relmae"],
                 result["metrics_overall"]["mae"],
                 result["metrics_overall"]["rmse"],
                 result["metrics_overall"]["dir_acc"])
    logging.info("Saved -> %s", out_dir)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="ARIF-Net training pipeline.")
    ap.add_argument("--model", required=True,
                    help="Nama model (persistence, arima, ridge, svr, rf, xgb, lgbm, "
                         "lstm, gru, bilstm, dialstm, tft, tcn_xgb) atau 'all'")
    ap.add_argument("--fold", default="F1", help="Fold validasi (F1/F2/F3) atau 'all'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Daftar model tabular (bisa langsung dilatih)
    tabular_models = ["persistence", "arima", "ridge", "svr", "rf", "xgb", "lgbm"]
    all_models = tabular_models + ["lstm", "gru", "bilstm", "dialstm", "tft", "tcn_xgb"]

    models = all_models if args.model == "all" else [args.model]
    folds = list(FOLDS) if args.fold == "all" else [args.fold]

    results_summary = []
    for model_name in models:
        for fold in folds:
            try:
                result = run_one(model_name, fold, args.seed)
                results_summary.append({
                    "model": model_name, "fold": fold, "seed": args.seed,
                    **result["metrics_overall"],
                })
            except Exception as exc:
                logging.error("GAGAL %s/%s: %s", model_name, fold, exc)
                results_summary.append({
                    "model": model_name, "fold": fold, "seed": args.seed,
                    "error": str(exc),
                })

    # Summary table
    if results_summary:
        summary = pd.DataFrame(results_summary)
        RESULTS.mkdir(parents=True, exist_ok=True)
        summary.to_csv(RESULTS / "training_summary.csv", index=False)
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
