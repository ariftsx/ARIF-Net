"""
ARIF-Net — Collector harga komoditas dari PIHPS Bank Indonesia.

Mengambil harga harian 3 komoditas (Cabai Merah Keriting, Bawang Merah, Beras Medium I)
dari Pasar Kramatjati level 3 (harga eceran).

Mode:
  smoke   Validasi comcat_id terhadap reference + ambil 1 chunk uji.
  full    Semua chunk bulanan 2019-01 … 2026-09 dengan resume otomatis.

Contoh:
  python -m 01_collection.collect_pihps smoke
  python -m 01_collection.collect_pihps full
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

from .utils import (
    CONFIG, DATA_RAW, HTTPClient, append_csv, load_json, load_project_config,
    month_chunks, now_utc, save_json, sha256_bytes, sha256_file, stamp,
    weekdays, write_new, setup_logging,
)

RAW = DATA_RAW / "pihps"
MANIFEST = RAW / "manifest.csv"
REFERENCE = RAW / "reference"
REPORTS = RAW / "reports"
LOGS = DATA_RAW.parent.parent / "logs"

MANIFEST_FIELDS = [
    "comcat_id", "commodity", "chunk", "start_date", "end_date",
    "request_url", "http_status", "raw_file", "raw_sha256", "raw_bytes",
    "n_date_keys", "n_reported", "n_dash", "n_expected_weekdays",
    "check", "error", "collected_at_utc",
]

# PIHPS BI settings (terverifikasi di project lama)
PIHPS_SETTINGS = {
    "base_url": "https://www.bi.go.id",
    "page_url": "https://www.bi.go.id/hargapangan",
    "data_url": "https://www.bi.go.id/hargapangan/WebSite/Home/GetGridData",
    "reference_url": "https://www.bi.go.id/hargapangan/WebSite/Home/GetRefFilter",
    "market_name": "Kramatjati",
    "market_level": 3,
    "delay_seconds": 1.5,
    "delay_jitter_seconds": 0.5,
    "timeout_seconds": 45,
    "retries": 4,
}


def build_params(comcat_id: str, start: date, end: date, ts: int) -> dict:
    """Parameter query untuk PIHPS BI API."""
    return {
        "price": comcat_id,
        "tpHarga": "3",         # level 3 = eceran
        "jnsPasar": "0",       # semua jenis pasar
        "startdate": start.strftime("%m/%d/%Y"),
        "enddate": end.strftime("%m/%d/%Y"),
        "area": "",
        "_": str(ts),
    }


def market_rows(payload: dict, market_name: str, level: int) -> list[dict]:
    """Cari row pasar target dalam response PIHPS."""
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"payload.data bukan list: {type(data)}")
    rows = []
    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        lvl = row.get("level", 0)
        if market_name.lower() in name.lower() and int(lvl) == level:
            rows.append(row)
    return rows


def date_items(row: dict) -> list[tuple[date, str]]:
    """Ekstrak pasangan (tanggal, nilai) dari row PIHPS."""
    items = []
    for k, v in row.items():
        try:
            d = date.fromisoformat(k)
            items.append((d, str(v)))
        except (ValueError, TypeError):
            continue
    return sorted(items, key=lambda x: x[0])


def parse_price(value: str) -> float | None:
    """Parse nilai harga; '-' = tidak dilaporkan."""
    v = value.strip().replace(",", "").replace(".", "")
    if v in ("-", "", "0"):
        return None
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"tidak bisa parse harga: '{value}'")


def norm_text(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s.strip().lower())


def check_chunk(content: bytes, start: date, end: date) -> tuple[list[str], dict]:
    """Validasi struktural chunk tanpa mengubah data."""
    try:
        payload = json.loads(content)
    except ValueError as exc:
        return [f"bukan JSON: {exc}"], {}
    try:
        rows = market_rows(payload, PIHPS_SETTINGS["market_name"], PIHPS_SETTINGS["market_level"])
    except ValueError as exc:
        return [str(exc)], {}
    if len(rows) != 1:
        return [f"row Kramatjati L3 = {len(rows)} (harus 1)"], {}
    items = date_items(rows[0])
    issues = []
    n_dash = n_rep = 0
    for d, v in items:
        try:
            p = parse_price(v)
            n_rep += p is not None
            n_dash += p is None
        except ValueError as exc:
            issues.append(f"{d}: {exc}")
    return issues, {
        "n_date_keys": len(items), "n_reported": n_rep, "n_dash": n_dash,
        "n_expected_weekdays": len(weekdays(start, end)),
    }


def done_chunks() -> set[tuple[str, str]]:
    """Chunk yang sudah OK di manifest."""
    out: set[tuple[str, str]] = set()
    if not MANIFEST.exists():
        return out
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["http_status"] == "200" and r["check"] == "OK":
                out.add((r["comcat_id"], r["chunk"]))
    return out


class PIHPSClient:
    """Session ke PIHPS BI dengan warm-up cookie."""

    def __init__(self):
        self.http = HTTPClient(
            delay=PIHPS_SETTINGS["delay_seconds"],
            retries=PIHPS_SETTINGS["retries"],
            timeout=PIHPS_SETTINGS["timeout_seconds"],
        )
        self._warmup()

    def _warmup(self):
        r = self.http.get(PIHPS_SETTINGS["page_url"], label="warmup")
        if r:
            self.http.session.headers.update({
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": PIHPS_SETTINGS["page_url"],
            })
            xsrf = self.http.session.cookies.get("XSRF-TOKEN")
            if xsrf:
                self.http.session.headers["XSRF-TOKEN"] = xsrf
            logging.info("PIHPS warm-up OK, cookies=%s", sorted(self.http.session.cookies.keys()))

    def fetch(self, url: str, params: dict, label: str):
        return self.http.get(url, params=params, label=label)


def cmd_smoke(args) -> int:
    setup_logging("pihps_smoke", LOGS)
    cfg = load_project_config()
    commodities = cfg["commodities"]
    client = PIHPSClient()

    # Fetch reference
    logging.info("Mengambil reference commodity dari PIHPS BI...")
    r = client.fetch(PIHPS_SETTINGS["reference_url"],
                     {"_": str(int(time.time() * 1000))}, "reference")
    if r and r.status_code == 200:
        REFERENCE.mkdir(parents=True, exist_ok=True)
        (REFERENCE / "ref_commodity.json").write_bytes(r.content)
        logging.info("Reference OK, disimpan.")

    # Test 1 chunk per komoditas
    start = date.fromisoformat(args.start) if args.start else date(2019, 1, 1)
    end = date.fromisoformat(args.end) if args.end else date(2019, 1, 31)
    results = []
    for comm in commodities:
        cid = comm["comcat_id"]
        params = build_params(cid, start, end, int(time.time() * 1000))
        r = client.fetch(PIHPS_SETTINGS["data_url"], params, f"smoke:{cid}")
        if r is None:
            logging.error("%s: gagal koneksi", cid)
            results.append({"comcat_id": cid, "status": "FAIL"})
            continue
        RAW.mkdir(parents=True, exist_ok=True)
        f = write_new(RAW / f"smoke_{cid}_{start:%Y-%m}.json", r.content)
        issues, stats = check_chunk(r.content, start, end) if r.status_code == 200 else (["HTTP"], {})
        ok = r.status_code == 200 and not issues
        results.append({"comcat_id": cid, "status": "PASS" if ok else "FAIL",
                        "stats": stats, "issues": issues})
        logging.info("[%s] %s: HTTP %s, tanggal=%s, dilaporkan=%s",
                     "PASS" if ok else "FAIL", cid, r.status_code,
                     stats.get("n_date_keys", 0), stats.get("n_reported", 0))

    verdict = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS / "smoke_pihps.json", {
        "step": "smoke PIHPS", "verdict": verdict, "results": results,
        "checked_at_utc": now_utc(),
    })
    logging.info("HASIL SMOKE PIHPS: %s", verdict)
    return 0 if verdict == "PASS" else 1


def cmd_full(args) -> int:
    setup_logging("pihps_full", LOGS)
    cfg = load_project_config()
    commodities = cfg["commodities"]
    start = date.fromisoformat(cfg["period"]["start"])
    end = date.fromisoformat(cfg["period"]["end"])
    chunks = month_chunks(start, end)
    done = done_chunks()
    todo = [(c, s, e) for c in commodities for s, e in chunks
            if (c["comcat_id"], f"{s:%Y-%m}") not in done]

    logging.info("Komoditas=%s | chunk=%s | sudah=%s | sisa=%s",
                 [c["comcat_id"] for c in commodities], len(chunks), len(done), len(todo))

    if not todo:
        logging.info("Semua chunk sudah selesai.")
        return 0

    client = PIHPSClient()
    n_ok = n_err = 0
    for comm, s, e in todo:
        cid = comm["comcat_id"]
        label = f"{cid}:{s:%Y-%m}"
        row = {"comcat_id": cid, "commodity": comm["pihps_name"],
               "chunk": f"{s:%Y-%m}", "start_date": str(s), "end_date": str(e),
               "collected_at_utc": now_utc()}
        try:
            r = client.fetch(PIHPS_SETTINGS["data_url"],
                             build_params(cid, s, e, int(time.time() * 1000)), label)
        except Exception as exc:
            row.update({"http_status": "EXC", "check": "EXCEPTION", "error": str(exc)[:300]})
            append_csv(MANIFEST, MANIFEST_FIELDS, row)
            n_err += 1
            logging.error("%s | gagal: %s", label, exc)
            continue

        if r is None:
            row.update({"http_status": "EXC", "check": "EXCEPTION", "error": "retries habis"})
            append_csv(MANIFEST, MANIFEST_FIELDS, row)
            n_err += 1
            continue

        f = write_new(RAW / f"{cid}_{s:%Y-%m}.json", r.content)
        issues, stats = check_chunk(r.content, s, e) if r.status_code == 200 else ([f"HTTP {r.status_code}"], {})
        row.update({
            "request_url": r.url, "http_status": r.status_code,
            "raw_file": str(f.relative_to(DATA_RAW.parent.parent)),
            "raw_sha256": sha256_bytes(r.content), "raw_bytes": len(r.content),
            **{k: stats.get(k, "") for k in ("n_date_keys", "n_reported", "n_dash", "n_expected_weekdays")},
            "check": "OK" if not issues else "FAIL",
            "error": "; ".join(issues)[:500],
        })
        append_csv(MANIFEST, MANIFEST_FIELDS, row)
        if issues:
            n_err += 1
            logging.error("%s | %s", label, row["error"])
        else:
            n_ok += 1
            logging.info("%-14s OK | tanggal=%s dilaporkan=%s", label,
                         stats["n_date_keys"], stats["n_reported"])

    logging.info("SELESAI: OK=%s ERROR=%s", n_ok, n_err)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Collector PIHPS ARIF-Net.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="Validasi + 1 chunk uji")
    sm.add_argument("--start", default=None)
    sm.add_argument("--end", default=None)
    sub.add_parser("full", help="Full collection (resume otomatis)")
    args = ap.parse_args()
    return cmd_smoke(args) if args.cmd == "smoke" else cmd_full(args)


if __name__ == "__main__":
    sys.exit(main())
