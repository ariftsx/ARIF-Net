"""
ARIF-Net — Collector harga komoditas dari PIHPS Bank Indonesia.

Mengambil harga harian 3 komoditas (Cabai Merah Keriting, Bawang Merah, Beras Medium I)
dari Pasar Kramatjati level 3 (harga eceran, pasar tradisional).

Endpoint, parameter, dan format respons di-porting dari collector project lama yang
sudah terbukti berhasil (Historical_Komoditas/PIHPS, smoke PASS 2026-10-03):
  * data       TabelHarga/GetGridDataKomoditas (bukan Home/GetGridData, yang hanya
               mengembalikan daftar survei)
  * kunci tgl  DD/MM/YYYY, harga "40,000" (koma = pemisah ribuan), "-" = tidak dilaporkan
  * market     baris "Pasar Kramatjati" level 3 di bawah DKI Jakarta (province 13, regency 34)

Mode:
  smoke   Verifikasi comcat_id terhadap reference + ambil 1 chunk uji per komoditas.
  full    Semua chunk bulanan pada periode project.json, resume otomatis.

Aturan: raw = evidence (byte apa adanya + sha256, tidak pernah ditimpa), tanpa imputasi.

Contoh:
  python -m 01_collection.collect_pihps smoke
  python -m 01_collection.collect_pihps full
  python -m 01_collection.collect_pihps full --comcat com_14
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from datetime import date, datetime

import requests

from .utils import (
    DATA_RAW, ROOT, HTTPClient, append_csv, load_project_config, month_chunks,
    now_utc, save_json, setup_logging, sha256_bytes, sha256_file, weekdays, write_new,
)

RAW = DATA_RAW / "pihps"
MANIFEST = RAW / "manifest.csv"
REFERENCE = RAW / "reference"
SMOKE = RAW / "smoke"
REPORTS = RAW / "reports"
LOGS = ROOT / "logs"

MANIFEST_FIELDS = [
    "comcat_id", "commodity", "chunk", "start_date", "end_date",
    "request_url", "http_status", "raw_file", "raw_sha256", "raw_bytes",
    "n_date_keys", "n_reported", "n_dash", "n_expected_weekdays",
    "check", "error", "collected_at_utc",
]

# PIHPS BI settings (terverifikasi di project lama, tangkapan DevTools peneliti)
PIHPS_SETTINGS = {
    "base_url": "https://www.bi.go.id/hargapangan",
    "page_url": "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas",
    "data_url": "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas",
    "reference_url": "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetRefCommodityAndCategory",
    "price_type_id": 1,         # pasar tradisional
    "province_id": 13,          # DKI Jakarta
    "regency_id": 34,           # label sumber "Kota Jakarta Pusat" (quirk; Kramatjati di Jaktim)
    "show_kota": "true",
    "show_pasar": "true",
    "tipe_laporan": 1,          # harian
    "market_name": "Pasar Kramatjati",
    "market_level": 3,
    "delay_seconds": 1.0,
    "delay_jitter_seconds": 0.5,
    "timeout_seconds": 60,
    "retries": 5,
}

DATE_KEY_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
PRICE_RE = re.compile(r"\d{1,3}(,\d{3})*(\.\d+)?|\d+(\.\d+)?")
MISSING_TOKENS = {"", "-", "null", "none"}


def build_params(comcat_id: str, start: date, end: date, ts: int) -> dict:
    """Parameter query GetGridDataKomoditas."""
    s = PIHPS_SETTINGS
    return {
        "price_type_id": s["price_type_id"],
        "comcat_id": comcat_id,
        "province_id": s["province_id"],
        "regency_id": s["regency_id"],
        "showKota": s["show_kota"],
        "showPasar": s["show_pasar"],
        "tipe_laporan": s["tipe_laporan"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "_": str(ts),
    }


def norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("-", " "))


def parse_price(value) -> float | None:
    """'50,000' -> 50000.0 ; '-' / '' -> None. Koma = pemisah ribuan."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    if not PRICE_RE.fullmatch(text):
        raise ValueError(f"format harga tidak dikenal: {text!r}")
    return float(text.replace(",", ""))


def market_rows(payload, market_name: str, level: int) -> list[dict]:
    """Baris pasar target (nama persis, level persis) dalam respons PIHPS."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("respons bukan {'data': [...]}")
    return [r for r in payload["data"] if isinstance(r, dict)
            and str(r.get("level")) == str(level)
            and norm_text(r.get("name")) == norm_text(market_name)]


def date_items(row: dict) -> list[tuple[date, str]]:
    """Kunci tanggal DD/MM/YYYY -> (date, nilai mentah sebagai string)."""
    out = []
    for k, v in row.items():
        if DATE_KEY_RE.match(str(k)):
            out.append((datetime.strptime(k, "%d/%m/%Y").date(), "" if v is None else str(v)))
    return sorted(out)


def check_chunk(content: bytes, start: date, end: date) -> tuple[list[str], dict]:
    """Validasi struktural chunk tanpa mengubah data."""
    s = PIHPS_SETTINGS
    try:
        payload = json.loads(content)
    except ValueError as exc:
        return [f"bukan JSON: {exc}"], {}
    try:
        rows = market_rows(payload, s["market_name"], s["market_level"])
    except ValueError as exc:
        return [str(exc)], {}
    if len(rows) != 1:
        names = [(r.get("level"), r.get("name")) for r in payload["data"] if isinstance(r, dict)]
        return [f"row '{s['market_name']}' L{s['market_level']} = {len(rows)} (harus 1); rows={names[:8]}"], {}
    items = date_items(rows[0])
    issues = []
    outside = [d for d, _ in items if not (start <= d <= end)]
    weekend = [d for d, _ in items if d.weekday() >= 5]
    if outside:
        issues.append(f"{len(outside)} tanggal di luar chunk")
    if weekend:
        issues.append(f"{len(weekend)} tanggal akhir pekan: {[d.isoformat() for d in weekend[:3]]}")
    n_dash = n_rep = 0
    for d, v in items:
        try:
            p = parse_price(v)
        except ValueError as exc:
            issues.append(f"{d}: {exc}")
            continue
        n_rep += p is not None
        n_dash += p is None
    return issues, {
        "n_date_keys": len(items), "n_reported": n_rep, "n_dash": n_dash,
        "n_expected_weekdays": len(weekdays(start, end)),
        "first_values": [(d.isoformat(), v) for d, v in items[:3]],
    }


def find_commodity_ids(payload, ids: set[str]) -> dict[str, list[str]]:
    """Cari nama untuk setiap comcat_id di respons reference. Return {comcat_id: [nama]}."""
    found: dict[str, list[str]] = {cid: [] for cid in ids}

    def walk(node) -> None:
        if isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            vals = [str(v).strip() for v in node.values() if isinstance(v, (str, int))]
            texts = [v for v in vals if not v.startswith(("com_", "cat_"))]
            for cid in (v for v in vals if v in ids):
                found[cid].extend(t for t in texts if norm_text(t) != norm_text(cid))
            for k, v in node.items():  # bentuk peta {"com_3": "Beras ..."}
                if k in ids and isinstance(v, str):
                    found[k].append(v)
            for v in node.values():
                walk(v)

    walk(payload)
    return found


def rel(path) -> str:
    return path.relative_to(ROOT).as_posix()


def done_chunks() -> set[tuple[str, str]]:
    """Chunk yang sudah OK di manifest dan file raw-nya masih utuh (sha256 cocok)."""
    out: set[tuple[str, str]] = set()
    if not MANIFEST.exists():
        return out
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = ROOT / r["raw_file"] if r["raw_file"] else None
            if (r["http_status"] == "200" and r["check"] == "OK" and p and p.exists()
                    and sha256_file(p) == r["raw_sha256"]):
                out.add((r["comcat_id"], r["chunk"]))
    return out


class PIHPSClient:
    """Session ke PIHPS BI: warm-up cookie, jeda + jitter, warm-up ulang bila sesi kedaluwarsa."""

    def __init__(self):
        self.http = HTTPClient(
            delay=PIHPS_SETTINGS["delay_seconds"],
            retries=PIHPS_SETTINGS["retries"],
            timeout=PIHPS_SETTINGS["timeout_seconds"],
        )
        self._warmup()

    def _warmup(self) -> None:
        h = self.http.session.headers
        for k in ("XSRF-TOKEN", "X-Requested-With"):
            h.pop(k, None)
        h.update({"Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                  "Referer": PIHPS_SETTINGS["base_url"]})
        r = self.http.session.get(PIHPS_SETTINGS["page_url"], timeout=PIHPS_SETTINGS["timeout_seconds"])
        r.raise_for_status()
        xsrf = self.http.session.cookies.get("XSRF-TOKEN")
        if xsrf:
            h["XSRF-TOKEN"] = xsrf
        h.update({"Accept": "application/json, text/javascript, */*; q=0.01",
                  "X-Requested-With": "XMLHttpRequest",
                  "Referer": PIHPS_SETTINGS["page_url"]})
        logging.info("PIHPS warm-up OK | HTTP %s | cookies=%s",
                     r.status_code, sorted(self.http.session.cookies.keys()))

    def fetch(self, url: str, params: dict, label: str) -> requests.Response | None:
        r = self.http.get(url, params=params, label=label)
        if r is not None and r.status_code == 200 and not r.content.lstrip().startswith(b"{"):
            logging.warning("%s: respons bukan JSON (sesi kedaluwarsa?) -> warm-up ulang", label)
            self._warmup()
            r = self.http.get(url, params=params, label=label)
        time.sleep(random.uniform(0, PIHPS_SETTINGS["delay_jitter_seconds"]))
        return r


def cmd_smoke(args) -> int:
    setup_logging("pihps_smoke", LOGS)
    commodities = load_project_config()["commodities"]
    names = {c["comcat_id"]: c["pihps_name"] for c in commodities}
    client = PIHPSClient()
    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        logging.info("[%s] %s%s", "PASS" if ok else "FAIL", name, f" — {detail}" if detail else "")

    # (a) reference -> verifikasi comcat_id (ID tidak boleh ditebak)
    r = client.fetch(PIHPS_SETTINGS["reference_url"], {"_": str(int(time.time() * 1000))}, "reference")
    id_check = {}
    ref_file = None
    if r is None:
        chk("reference HTTP 200", False, "retries habis")
    else:
        ref_file = write_new(REFERENCE / "ref_commodity.json", r.content)
        chk("reference HTTP 200", r.status_code == 200, f"status={r.status_code} file={rel(ref_file)}")
    if r is not None and r.status_code == 200:
        try:
            found = find_commodity_ids(json.loads(r.content), set(names))
        except ValueError as exc:
            found = {}
            chk("reference = JSON", False, str(exc))
        for cid, name in names.items():
            hits = sorted(set(found.get(cid, [])))
            ok = any(norm_text(h) == norm_text(name) for h in hits)
            id_check[cid] = {"expected": name, "found_names": hits[:5], "verified": ok}
            chk(f"comcat_id {cid} = '{name}' di reference", ok, f"ditemukan: {hits[:3]}")

    # (b) 1 chunk uji per komoditas
    chunk_report = {}
    for cid, name in names.items():
        r = client.fetch(PIHPS_SETTINGS["data_url"],
                         build_params(cid, args.start, args.end, int(time.time() * 1000)), f"smoke:{cid}")
        if r is None:
            chk(f"{cid} HTTP 200", False, "retries habis")
            chunk_report[cid] = {"commodity": name, "issues": ["retries habis"]}
            continue
        f = write_new(SMOKE / f"{cid}_{args.start:%Y-%m}.json", r.content)
        chk(f"{cid} HTTP 200", r.status_code == 200, f"status={r.status_code}")
        issues, stats = check_chunk(r.content, args.start, args.end) if r.status_code == 200 else (["HTTP"], {})
        chk(f"{cid} struktur & row {PIHPS_SETTINGS['market_name']} L{PIHPS_SETTINGS['market_level']}",
            not issues, "; ".join(issues) or
            f"tanggal={stats['n_date_keys']} (hari kerja {stats['n_expected_weekdays']}) "
            f"dilaporkan={stats['n_reported']} '-'={stats['n_dash']} contoh={stats['first_values']}")
        chunk_report[cid] = {"commodity": name, "request_url": r.url, "http_status": r.status_code,
                             "raw_file": rel(f), "raw_sha256": sha256_bytes(r.content),
                             "issues": issues, **stats}

    verdict = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    save_json(REPORTS / "smoke_pihps.json", {
        "step": "smoke PIHPS", "verdict": verdict, "period": [str(args.start), str(args.end)],
        "reference_file": rel(ref_file) if ref_file else None, "id_verification": id_check,
        "chunks": chunk_report, "checks": checks, "checked_at_utc": now_utc(),
    })
    logging.info("HASIL SMOKE PIHPS: %s | laporan: %s", verdict, rel(REPORTS / "smoke_pihps.json"))
    return 0 if verdict == "PASS" else 1


def cmd_full(args) -> int:
    setup_logging("pihps_full", LOGS)
    cfg = load_project_config()
    commodities = cfg["commodities"]
    start = date.fromisoformat(cfg["period"]["start"])
    end = date.fromisoformat(cfg["period"]["end"])
    chunks = month_chunks(start, end)
    targets = [c for c in commodities if not args.comcat or c["comcat_id"] == args.comcat]
    done = done_chunks()
    todo = [(c, s, e) for c in targets for s, e in chunks
            if (c["comcat_id"], f"{s:%Y-%m}") not in done]

    logging.info("Komoditas=%s | chunk/komoditas=%s | sudah=%s | sisa=%s",
                 [c["comcat_id"] for c in targets], len(chunks), len(done), len(todo))

    client = PIHPSClient() if todo else None
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
        except requests.RequestException as exc:
            r, err = None, str(exc)[:300]
        else:
            err = "retries habis"
        if r is None:
            row.update({"http_status": "EXC", "check": "EXCEPTION", "error": err})
            append_csv(MANIFEST, MANIFEST_FIELDS, row)
            n_err += 1
            logging.error("%s | gagal: %s", label, err)
            continue

        suffix = ".json" if r.status_code == 200 else ".err.json"
        f = write_new(RAW / f"{cid}_{s:%Y-%m}{suffix}", r.content)
        issues, stats = (check_chunk(r.content, s, e) if r.status_code == 200
                         else ([f"HTTP {r.status_code}"], {}))
        row.update({
            "request_url": r.url, "http_status": r.status_code, "raw_file": rel(f),
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
            logging.info("%-14s OK | tanggal=%2s/%2s hari kerja | dilaporkan=%2s | '-'=%s", label,
                         stats["n_date_keys"], stats["n_expected_weekdays"],
                         stats["n_reported"], stats["n_dash"])

    done_after = done_chunks()
    expected = {(c["comcat_id"], f"{s:%Y-%m}") for c in commodities for s, _ in chunks}
    per = {c["comcat_id"]: sum(1 for k in done_after if k[0] == c["comcat_id"]) for c in commodities}
    complete = expected <= done_after
    save_json(RAW / "collection_summary.json", {
        "step": "full collection PIHPS", "market": PIHPS_SETTINGS["market_name"],
        "market_level": PIHPS_SETTINGS["market_level"], "start_date": str(start), "end_date": str(end),
        "expected_chunks": len(expected), "completed_chunks_total": len(expected & done_after),
        "completed_chunks_per_commodity": per,
        "this_run": {"requests_ok": n_ok, "requests_error": n_err},
        "verdict": "PASS" if complete else "INCOMPLETE",
        "manifest": rel(MANIFEST), "finished_at_utc": now_utc(),
    })
    logging.info("SELESAI: %s | %s/%s chunk %s | run ini OK=%s ERROR=%s",
                 "PASS" if complete else "INCOMPLETE", len(expected & done_after), len(expected),
                 per, n_ok, n_err)
    if n_err:
        logging.info("Ada chunk gagal -> jalankan ulang perintah yang sama (chunk sukses di-skip).")
    return 0 if complete else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Collector PIHPS ARIF-Net.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="Verifikasi comcat_id + 1 chunk uji per komoditas")
    sm.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    sm.add_argument("--end", type=date.fromisoformat, default=date(2019, 1, 31))
    fu = sub.add_parser("full", help="Full collection (resume otomatis)")
    fu.add_argument("--comcat", default=None, help="Batasi ke satu comcat_id")
    args = ap.parse_args()
    return cmd_smoke(args) if args.cmd == "smoke" else cmd_full(args)


if __name__ == "__main__":
    sys.exit(main())
