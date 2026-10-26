"""
ARIF-Net -- Preprocessing harga komoditas PIHPS.

Membaca raw JSON chunks dari collector PIHPS -> normalisasi ke DataFrame harian
per komoditas, satu baris = satu tanggal, satu kolom = harga (Rp).

Sumber chunk = manifest collector: per (comcat_id, bulan) diambil entri OK terbaru,
sha256 raw diverifikasi. Parsing (kunci DD/MM/YYYY, "40,000", "-") memakai fungsi
collector agar format PIHPS hanya didefinisikan di satu tempat.
Nilai "-" = tidak dilaporkan -> tidak ditulis (tanpa imputasi/ffill).

Output: data/processed/pihps/prices_{comcat_id}.csv
        data/processed/pihps/prices_combined.csv
        data/processed/pihps/preprocess_pihps_report.json

Contoh:
  python -m 02_preprocessing.preprocess_pihps
"""
from __future__ import annotations

import json
import logging
import sys
from importlib import import_module

import pandas as pd

from .utils import DATA_PROCESSED, ROOT, load_project_config, now_utc, save_json, sha256_file

collector = import_module("01_collection.collect_pihps")

OUT = DATA_PROCESSED / "pihps"


def ok_chunks() -> pd.DataFrame:
    """Entri manifest sukses terbaru per (comcat_id, chunk)."""
    man = pd.read_csv(collector.MANIFEST, dtype=str, keep_default_na=False)
    ok = man[(man["http_status"] == "200") & (man["check"] == "OK")]
    return (ok.sort_values("collected_at_utc")
              .drop_duplicates(["comcat_id", "chunk"], keep="last")
              .sort_values(["comcat_id", "chunk"]))


def build_prices(chunks: pd.DataFrame, comcat_id: str, problems: list[str]) -> tuple[pd.DataFrame, int]:
    """Gabungkan raw chunks satu komoditas -> (DataFrame (date, price), jumlah '-')."""
    s = collector.PIHPS_SETTINGS
    records, n_dash = [], 0
    for _, r in chunks[chunks["comcat_id"] == comcat_id].iterrows():
        path = ROOT / r["raw_file"]
        if not path.exists():
            problems.append(f"file hilang: {r['raw_file']}")
            continue
        if sha256_file(path) != r["raw_sha256"]:
            problems.append(f"sha256 tidak cocok (raw berubah?): {r['raw_file']}")
            continue
        rows = collector.market_rows(json.loads(path.read_bytes()), s["market_name"], s["market_level"])
        if len(rows) != 1:
            problems.append(f"row market != 1 di {r['raw_file']}")
            continue
        for d, raw in collector.date_items(rows[0]):
            price = collector.parse_price(raw)
            if price is None:
                n_dash += 1
            elif d.weekday() < 5:  # Hanya hari kerja
                records.append({"date": d.isoformat(), "price": price})
    if not records:
        return pd.DataFrame(columns=["date", "price"]), n_dash
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    dup = int(df.duplicated(subset="date").sum())
    if dup:
        problems.append(f"{comcat_id}: {dup} tanggal duplikat antar chunk")
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df, n_dash


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_project_config()
    if not collector.MANIFEST.exists():
        logging.error("Manifest PIHPS tidak ada (%s). Jalankan collect_pihps full dulu.", collector.MANIFEST)
        return 1

    chunks = ok_chunks()
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    per: dict[str, dict] = {}
    all_frames = []

    for comm in cfg["commodities"]:
        cid = comm["comcat_id"]
        logging.info("Processing PIHPS %s (%s)...", cid, comm["name"])
        df, n_dash = build_prices(chunks, cid, problems)
        per[cid] = {"chunks": int((chunks["comcat_id"] == cid).sum()), "rows": len(df), "not_reported_dash": n_dash}
        if df.empty:
            logging.warning("%s: tidak ada data!", cid)
            problems.append(f"{cid}: tidak ada data")
            continue
        df.insert(0, "comcat_id", cid)
        df.insert(1, "commodity", comm["name"])

        out_path = OUT / f"prices_{cid}.csv"
        df.to_csv(out_path, index=False)
        per[cid].update({"start": str(df["date"].min().date()), "end": str(df["date"].max().date())})
        logging.info("%s: %d baris, %s -> %s, harga %.0f-%.0f, '-'=%d",
                     cid, len(df), df["date"].min().date(), df["date"].max().date(),
                     df["price"].min(), df["price"].max(), n_dash)
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUT / "prices_combined.csv", index=False)
        logging.info("Combined: %d baris -> prices_combined.csv", len(combined))

    verdict = "PASS" if not problems else "FAIL"
    for p in problems:
        logging.error(p)
    save_json(OUT / "preprocess_pihps_report.json", {
        "step": "preprocess_pihps", "verdict": verdict, "commodities": per,
        "problems": problems, "finished_at_utc": now_utc(),
    })
    logging.info("HASIL PREPROCESS PIHPS: %s", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
