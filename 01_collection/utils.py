"""
ARIF-Net -- Utilitas bersama untuk seluruh pipeline data collection.

Menyediakan: HTTP client dengan retry & rate-limit, logging, path helpers,
hashing, waktu WIB, keyword matching, dan I/O.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

# ────────────────────────────────────────────── paths
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FINAL = ROOT / "data" / "final"
RESULTS = ROOT / "results"

WIB = timezone(timedelta(hours=7))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36")

MONTHS_ID = {
    "jan": 1, "januari": 1, "feb": 2, "februari": 2, "mar": 3, "maret": 3,
    "apr": 4, "april": 4, "mei": 5, "jun": 6, "juni": 6, "jul": 7, "juli": 7,
    "agu": 8, "ags": 8, "agustus": 8, "agt": 8, "sep": 9, "september": 9,
    "okt": 10, "oktober": 10, "nov": 11, "november": 11, "des": 12, "desember": 12,
}


# ────────────────────────────────────────────── config I/O
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_project_config() -> dict:
    return load_json(CONFIG / "project.json")


def load_news_config() -> tuple[dict, dict]:
    return load_json(CONFIG / "news_sources.json"), load_json(CONFIG / "news_scope.json")


def load_locations() -> dict:
    return load_json(CONFIG / "locations.json")


# ────────────────────────────────────────────── time & date
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def daterange(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def weekdays(start: date, end: date) -> list[date]:
    return [d for d in daterange(start, end) if d.weekday() < 5]


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Generate (first_day, last_day) per bulan antara start dan end."""
    chunks = []
    d = start.replace(day=1)
    while d <= end:
        last = (d.replace(month=d.month % 12 + 1, day=1) - timedelta(days=1)
                if d.month < 12 else d.replace(month=12, day=31))
        s = max(d, start)
        e = min(last, end)
        chunks.append((s, e))
        d = last + timedelta(days=1)
    return chunks


def to_wib_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=WIB)
    return dt.astimezone(WIB).strftime("%Y-%m-%dT%H:%M:%S+07:00")


def parse_iso_any(text: str) -> datetime | None:
    """ISO 8601 dengan zona (+00:00, +07:00, Z). Tanpa zona -> None."""
    t = (text or "").strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    return dt if dt.tzinfo else None


def parse_id_datetime(text: str) -> datetime | None:
    """'Rabu, 02 Jan 2019 22:00 WIB' -> datetime WIB."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})(?:\D+(\d{1,2})[:.](\\d{2}))?", text or "")
    if not m or m.group(2).lower() not in MONTHS_ID:
        return None
    hh, mi = (int(m.group(4)), int(m.group(5))) if m.group(4) else (0, 0)
    return datetime(int(m.group(3)), MONTHS_ID[m.group(2).lower()], int(m.group(1)), hh, mi, tzinfo=WIB)


def kompas_url_time(url: str) -> datetime | None:
    """/read/YYYY/MM/DD/HHMMSSxxx/ -> WIB."""
    m = re.search(r"/read/(\d{4})/(\d{2})/(\d{2})/(\d{2})(\d{2})(\d{2})\d*", url)
    if not m:
        return None
    y, mo, d, hh, mi, ss = map(int, m.groups())
    if hh > 23 or mi > 59 or ss > 59:
        return None
    return datetime(y, mo, d, hh, mi, ss, tzinfo=WIB)


# ────────────────────────────────────────────── hashing & raw I/O
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_raw_gz(path: Path, data: bytes) -> Path:
    """Simpan bytes response ter-gzip; tidak pernah menimpa file yang ada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target, n = path, 1
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}.r{n}.gz")
    with target.open("xb") as fh:
        fh.write(gzip.compress(data, mtime=0))
    return target


def read_raw_gz(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def write_new(path: Path, data: bytes | str) -> Path:
    """Tulis file baru; jika sudah ada, tambah suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target, n = path, 1
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}.v{n}{path.suffix}")
    if isinstance(data, bytes):
        target.write_bytes(data)
    else:
        target.write_text(data, encoding="utf-8")
    return target


# ────────────────────────────────────────────── CSV manifest
def append_csv(path: Path, fields: list[str], row: dict) -> None:
    """Append satu baris ke CSV manifest (thread-safe via file lock)."""
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ────────────────────────────────────────────── URL helpers
def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit(("https", parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def article_id(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()


def build_index_url(template: str, d: date, page: int, per_page: int) -> str:
    """Bangun URL indeks dari template dengan token tanggal & paginasi."""
    tokens = {
        "date_iso": d.isoformat(),
        "date_next_iso": (d + timedelta(days=1)).isoformat(),
        "date_mdy_enc": f"{d:%m}%2F{d:%d}%2F{d:%Y}",
        "date_ymd_slash": f"{d:%Y/%m/%d}",
        "yyyy": f"{d:%Y}", "mm": f"{d:%m}", "dd": f"{d:%d}", "d": str(d.day),
        "page": str(page), "offset": str((page - 1) * per_page),
    }
    return template.format(**tokens)


# ────────────────────────────────────────────── HTTP client
class HTTPClient:
    """HTTP session dengan retry, backoff, rate-limiting, dan warm-up."""

    def __init__(self, delay: float = 1.0, retries: int = 4, timeout: int = 45):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Accept-Language": "id-ID,id;q=0.9",
        })
        self.delay = delay
        self.retries = retries
        self.timeout = timeout

    def get(self, url: str, params: dict | None = None,
            label: str = "") -> requests.Response | None:
        log = logging.getLogger("http")
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("%s koneksi gagal (%s/%s): %s", label, attempt, self.retries, exc)
                time.sleep(10 * attempt)
                continue
            time.sleep(self.delay)
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("%s HTTP %s (%s/%s) -> tunggu %ss",
                            label, r.status_code, attempt, self.retries, 30 * attempt)
                time.sleep(30 * attempt)
                continue
            return r
        return None


# ────────────────────────────────────────────── keyword matching (relevansi berita)
class KeywordMatcher:
    """Cocokkan kata kunci pada teks (case-insensitive, batas kata)."""

    def __init__(self, topics: dict[str, list[str]]):
        self.rules = [
            (topic, kw, re.compile(r"(?<![0-9a-z])" + re.escape(kw.lower()) + r"(?![0-9a-z])"))
            for topic, keywords in topics.items()
            for kw in keywords
        ]

    def match(self, text: str) -> tuple[list[str], list[str]]:
        """Return (topics_matched, keywords_matched)."""
        low = (text or "").lower()
        hits = [(t, kw) for t, kw, rx in self.rules if rx.search(low)]
        return sorted({t for t, _ in hits}), sorted({kw for _, kw in hits})

    def is_relevant(self, text: str) -> bool:
        topics, _ = self.match(text)
        return bool(topics)


class RegionMatcher:
    """Cocokkan nama kabupaten pemasok di teks."""

    def __init__(self, locations: list[dict]):
        names = sorted(
            ((loc["kabupaten"].replace("Kabupaten ", "").lower(), loc["location_id"])
             for loc in locations),
            key=lambda x: -len(x[0])
        )
        self.rules = [
            (lid, re.compile(r"(?<![0-9a-z])" + re.escape(n) + r"(?![0-9a-z])"))
            for n, lid in names
        ]

    def match(self, title: str) -> list[str]:
        low, hits = (title or "").lower(), []
        for lid, rx in self.rules:
            if rx.search(low):
                hits.append(lid)
                low = rx.sub(" ", low)
        return sorted(hits)


# ────────────────────────────────────────────── logging setup
def setup_logging(name: str, log_dir: Path | None = None) -> Path:
    if log_dir is None:
        log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{name}_{stamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-15s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile, encoding="utf-8"),
        ],
    )
    logging.info("Log: %s", logfile)
    return logfile
