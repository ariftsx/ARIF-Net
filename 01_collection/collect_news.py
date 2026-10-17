"""
ARIF-Net -- Collector Berita dari 3 Media (Kompas, Detik, CNN Indonesia).

PERUBAHAN dari project lama:
  1. Mengambil HEADLINE + ISI BERITA (bukan hanya headline)
  2. Filter relevansi SEBELUM scraping body (hemat bandwidth)
  3. Menggunakan trafilatura untuk ekstraksi body text bersih
  4. Output: CSV dengan kolom title, body_text, published_at, topics, dll.

Pipeline:
  Tahap 1: Scrape halaman indeks -> daftar URL + judul
  Tahap 2: Filter relevansi berdasarkan judul (keyword matching)
  Tahap 3: Scrape isi berita relevan (trafilatura)
  Tahap 4: Normalisasi -> articles_relevant.csv

Contoh:
  python -m 01_collection.collect_news index --source kompas --max-minutes 60
  python -m 01_collection.collect_news articles --max-minutes 120
  python -m 01_collection.collect_news normalize
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

from .utils import (
    CONFIG, DATA_RAW, HTTPClient, KeywordMatcher, RegionMatcher,
    append_csv, article_id, build_index_url, canonical_url, daterange,
    kompas_url_time, load_json, load_locations, load_news_config, load_project_config,
    now_utc, parse_id_datetime, parse_iso_any, save_json, save_raw_gz,
    setup_logging, sha256_bytes, stamp, to_wib_iso,
)

RAW = DATA_RAW / "news"
RAW_INDEX = RAW / "index_pages"
RAW_ARTICLES = RAW / "articles"
PROCESSED = DATA_RAW.parent / "processed" / "news"
REPORTS = RAW / "reports"
LOGS = DATA_RAW.parent.parent / "logs"

MANIFEST_INDEX = RAW / "manifest_index.csv"
MANIFEST_ARTICLES = RAW / "manifest_articles.csv"

INDEX_FIELDS = [
    "source", "index", "date", "page", "url", "http_status",
    "n_items", "n_relevant", "raw_sha256", "collected_at_utc",
]
ARTICLE_FIELDS = [
    "article_id", "source", "url", "title", "published_at_wib",
    "body_length", "topics_matched", "keywords_matched", "regions_matched",
    "http_status", "extract_method", "check", "collected_at_utc",
]


# ──────────────────────────────────────── HTML Parsers (indeks per media)
def _txt(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip() if node else ""


def _item(url, title, channel="", time_wib=None, source="", raw_time=""):
    return {"url": url.strip(), "title": title.strip(), "channel": channel.strip(),
            "time_wib": time_wib, "time_source": source if time_wib else "",
            "index_time_text": raw_time.strip()}


def parse_kompas_index(html: bytes) -> list[dict]:
    s = BeautifulSoup(html, "lxml")
    out = []
    for it in s.select("div.articleItem"):
        a = it.select_one("a.article-link[href]")
        if not a or "/read/" not in a["href"]:
            continue
        out.append(_item(a["href"], _txt(it.select_one(".articleTitle")),
                         _txt(it.select_one(".articlePost-subtitle")),
                         kompas_url_time(a["href"]), "url",
                         _txt(it.select_one(".articlePost-date"))))
    return out


def parse_detik_index(html: bytes) -> list[dict]:
    s = BeautifulSoup(html, "lxml")
    out = []
    for it in s.select("article.list-content__item"):
        a = it.select_one("a.media__link[href]") or it.select_one("a[href*='/d-']")
        if not a:
            continue
        t = it.select_one("[title*='WIB']")
        raw = t.get("title", "") if t else ""
        title = _txt(it.select_one(".media__title")) or (it.find("img") or {}).get("alt", "")
        out.append(_item(a["href"], title, a["href"].split("/")[3] if a["href"].count("/") > 3 else "",
                         parse_id_datetime(raw), "index", raw))
    return out


def parse_cnn_index(html: bytes) -> list[dict]:
    s = BeautifulSoup(html, "lxml")
    out = []
    for it in s.find_all("article"):
        a = it.find("a", href=re.compile(r"^https://www\.cnnindonesia\.com/[a-z\-]+/\d{14}-\d+-\d+/"))
        h2 = it.find("h2")
        if not a or not h2:
            continue
        chan = a["href"].split("/")[3]
        out.append(_item(a["href"], _txt(h2), chan))
    return out


INDEX_PARSERS = {
    "kompas": parse_kompas_index,
    "detik": parse_detik_index,
    "cnnindonesia": parse_cnn_index,
}


# ──────────────────────────────────────── Body Text Extractor
def extract_body(html_content: bytes, url: str) -> tuple[str, str]:
    """Ekstrak isi berita menggunakan trafilatura (prioritas) + fallback BeautifulSoup.

    Returns: (body_text, method_used)
    """
    try:
        import trafilatura
        result = trafilatura.extract(
            html_content.decode("utf-8", errors="replace"),
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        if result and len(result) > 100:
            return result.strip(), "trafilatura"
    except Exception:
        pass

    # Fallback: BeautifulSoup
    try:
        soup = BeautifulSoup(html_content, "lxml")
        # Hapus elemen tidak relevan
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside",
                                  "figure", "figcaption", "iframe"]):
            tag.decompose()
        # Cari konten utama berdasarkan class/id umum media Indonesia
        selectors = [
            "div.read__content", "div.detail__body-text",           # Kompas, Detik
            "div.content-detail", "div.detail-text",                # CNN ID
            "article .entry-content", "div[itemprop='articleBody']",
            "div.article__content", "div.post-content",
        ]
        for sel in selectors:
            content = soup.select_one(sel)
            if content:
                # Hapus elemen "Baca juga"
                for baca in content.find_all(["strong", "b", "a"],
                                             string=re.compile(r"baca juga|baca:", re.I)):
                    parent = baca.find_parent(["p", "div"])
                    if parent:
                        parent.decompose()
                text = _txt(content)
                if len(text) > 100:
                    return text, "bs4_selector"

        # Fallback: ambil semua <p>
        paragraphs = soup.find_all("p")
        text = " ".join(_txt(p) for p in paragraphs if len(_txt(p)) > 30)
        if text and len(text) > 100:
            return text, "bs4_paragraphs"
    except Exception:
        pass

    return "", "failed"


def extract_article_time(html: bytes) -> tuple[datetime | None, str]:
    """Waktu terbit dari halaman artikel: JSON-LD -> meta tag."""
    text = html.decode("utf-8", "replace")
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', text, flags=re.S):
        try:
            data = json.loads(block)
        except ValueError:
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', block)
            data = {"datePublished": m.group(1)} if m else {}
        for d in (data if isinstance(data, list) else [data]):
            dt = parse_iso_any(str(d.get("datePublished", ""))) if isinstance(d, dict) else None
            if dt:
                return dt, "jsonld"
    m = re.search(r'<meta[^>]+article:published_time[^>]+content="([^"]+)"', text)
    if m and parse_iso_any(m.group(1)):
        return parse_iso_any(m.group(1)), "meta"
    return None, ""


# ──────────────────────────────────────── Tahap 1: Index Collection
def cmd_index(args) -> int:
    """Scrape halaman indeks per (media, tanggal) -- daftar URL + judul."""
    setup_logging("news_index", LOGS)
    news_cfg, scope = load_news_config()
    proj_cfg = load_project_config()

    sources = list(news_cfg["sources"]) if args.source == "all" else args.source.split(",")
    matcher = KeywordMatcher(scope["topics"])

    if args.dates:
        dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]
    else:
        dates = daterange(
            date.fromisoformat(args.start or proj_cfg["period"]["start"]),
            date.fromisoformat(args.end or proj_cfg["period"]["end"]),
        )

    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    http = HTTPClient(delay=news_cfg["delay_seconds"], retries=news_cfg["retries"],
                      timeout=news_cfg["timeout_seconds"])

    n_total = n_relevant = 0
    for source in sources:
        scfg = news_cfg["sources"][source]
        parser = INDEX_PARSERS[source]
        for idx in scfg["indexes"]:
            for d in dates:
                if deadline and time.time() > deadline:
                    logging.info("Batas waktu tercapai, berhenti.")
                    return 0
                page = 1
                while page <= news_cfg["max_pages_per_day"]:
                    url = build_index_url(idx["url"], d, page, scfg["per_page"])
                    r = http.get(url, label=f"{source}/{idx['name']}/{d}/p{page}")
                    if r is None or r.status_code != 200:
                        break
                    items = parser(r.content)
                    if not items:
                        break
                    relevant = [it for it in items if matcher.is_relevant(it["title"])]
                    n_total += len(items)
                    n_relevant += len(relevant)
                    append_csv(MANIFEST_INDEX, INDEX_FIELDS, {
                        "source": source, "index": idx["name"], "date": d.isoformat(),
                        "page": page, "url": url, "http_status": r.status_code,
                        "n_items": len(items), "n_relevant": len(relevant),
                        "raw_sha256": sha256_bytes(r.content), "collected_at_utc": now_utc(),
                    })

                    # Simpan item relevan ke CSV sementara untuk Tahap 2
                    items_file = RAW / "items" / f"{source}_{d.isoformat()}_{idx['name']}_p{page}.json"
                    items_file.parent.mkdir(parents=True, exist_ok=True)
                    save_json(items_file, [
                        {**it, "source": source, "index": idx["name"],
                         "index_date": d.isoformat(), "page": page,
                         "article_id": article_id(it["url"]),
                         "time_wib": to_wib_iso(it["time_wib"]) if it["time_wib"] else None}
                        for it in items
                    ])
                    logging.info("%s %s %s p%d: %d item, %d relevan",
                                 source, idx["name"], d, page, len(items), len(relevant))
                    page += 1

    logging.info("INDEX SELESAI: total=%d relevan=%d", n_total, n_relevant)
    return 0


# ──────────────────────────────────────── Tahap 2: Article Scraping
def iter_relevant_items(sources: list[str] | None = None) -> list[dict]:
    """Baca semua item dari file JSON items/, filter yang relevan."""
    items_dir = RAW / "items"
    if not items_dir.exists():
        return []
    _, scope = load_news_config()
    matcher = KeywordMatcher(scope["topics"])
    all_items = []
    seen = set()
    for f in sorted(items_dir.glob("*.json")):
        items = load_json(f)
        for it in items:
            if sources and it.get("source") not in sources:
                continue
            aid = it.get("article_id", article_id(it["url"]))
            if aid in seen:
                continue
            seen.add(aid)
            if matcher.is_relevant(it["title"]):
                it["article_id"] = aid
                all_items.append(it)
    return all_items


def cmd_articles(args) -> int:
    """Scrape isi berita relevan (headline sudah cocok keyword)."""
    setup_logging("news_articles", LOGS)
    news_cfg, scope = load_news_config()
    locs = load_locations()
    matcher = KeywordMatcher(scope["topics"])
    region_matcher = RegionMatcher(locs["locations"])

    # Cari artikel yang sudah di-scrape
    done = set()
    if MANIFEST_ARTICLES.exists():
        with MANIFEST_ARTICLES.open(encoding="utf-8") as fh:
            done = {r["article_id"] for r in csv.DictReader(fh) if r["check"] == "OK"}

    items = iter_relevant_items()
    todo = [it for it in items if it["article_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    logging.info("Artikel relevan: %d total, %d sudah, %d sisa", len(items), len(done), len(todo))

    if not todo:
        logging.info("Tidak ada artikel baru untuk di-scrape.")
        return 0

    http = HTTPClient(delay=news_cfg["delay_seconds"], retries=news_cfg["retries"],
                      timeout=news_cfg["timeout_seconds"])
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    n_ok = n_fail = 0

    for it in todo:
        if deadline and time.time() > deadline:
            logging.info("Batas waktu tercapai, berhenti.")
            break

        r = http.get(it["url"], label=f"art:{it['article_id'][:8]}")
        row = {"article_id": it["article_id"], "source": it.get("source", ""),
               "url": it["url"], "title": it["title"], "collected_at_utc": now_utc()}

        if r is None:
            row.update({"http_status": "EXC", "check": "FAIL"})
            append_csv(MANIFEST_ARTICLES, ARTICLE_FIELDS, row)
            n_fail += 1
            continue

        # Simpan raw article
        raw_path = RAW_ARTICLES / it.get("source", "unknown") / f"{it['article_id'][:16]}.gz"
        save_raw_gz(raw_path, r.content)

        # Ekstrak body text
        body_text, method = extract_body(r.content, it["url"]) if r.status_code == 200 else ("", "failed")

        # Simpan body text ke file terpisah (untuk preprocessing)
        if body_text:
            text_dir = PROCESSED / "article_texts"
            text_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / f"{it['article_id']}.txt").write_text(body_text, encoding="utf-8")

        # Waktu terbit
        pub_time = it.get("time_wib", "")
        if not pub_time and r.status_code == 200:
            dt, src = extract_article_time(r.content)
            pub_time = to_wib_iso(dt) if dt else ""

        # Relevansi (recheck dengan body text juga)
        full_text = it["title"] + " " + body_text
        topics, kws = matcher.match(full_text)
        regions = region_matcher.match(full_text)

        check = "OK" if body_text and len(body_text) > 50 else "NO_BODY"
        row.update({
            "published_at_wib": pub_time,
            "body_length": len(body_text),
            "topics_matched": "|".join(topics),
            "keywords_matched": "|".join(kws),
            "regions_matched": "|".join(regions),
            "http_status": r.status_code,
            "extract_method": method,
            "check": check,
        })
        append_csv(MANIFEST_ARTICLES, ARTICLE_FIELDS, row)

        if check == "OK":
            n_ok += 1
        else:
            n_fail += 1
        logging.info("[%s] %s | body=%d chars | method=%s | topics=%s",
                     check, it["article_id"][:8], len(body_text), method, topics)

    logging.info("ARTICLES SELESAI: OK=%d FAIL=%d", n_ok, n_fail)
    return 0


# ──────────────────────────────────────── Tahap 3: Normalize
def cmd_normalize(args) -> int:
    """Normalisasi: gabungkan index items + article bodies -> articles_relevant.csv."""
    setup_logging("news_normalize", LOGS)
    _, scope = load_news_config()
    matcher = KeywordMatcher(scope["topics"])
    locs = load_locations()
    region_matcher = RegionMatcher(locs["locations"])

    # Baca manifest articles
    articles = {}
    if MANIFEST_ARTICLES.exists():
        with MANIFEST_ARTICLES.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["check"] == "OK":
                    articles[r["article_id"]] = r

    # Baca semua items
    items = iter_relevant_items()
    logging.info("Items relevan: %d, articles scraped OK: %d", len(items), len(articles))

    # Gabungkan
    COLS = [
        "article_id", "source", "index", "channel", "url", "title",
        "published_at_wib", "body_length", "body_preview",
        "topics_matched", "keywords_matched", "regions_matched",
        "index_date", "is_relevant",
    ]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    text_dir = PROCESSED / "article_texts"
    rows = []
    for it in items:
        aid = it["article_id"]
        art = articles.get(aid, {})

        # Baca body text jika ada
        body = ""
        body_file = text_dir / f"{aid}.txt"
        if body_file.exists():
            body = body_file.read_text(encoding="utf-8")

        full_text = it["title"] + " " + body
        topics, kws = matcher.match(full_text)
        regions = region_matcher.match(full_text)

        rows.append({
            "article_id": aid,
            "source": it.get("source", ""),
            "index": it.get("index", ""),
            "channel": it.get("channel", ""),
            "url": it["url"],
            "title": it["title"],
            "published_at_wib": art.get("published_at_wib", it.get("time_wib", "")),
            "body_length": len(body),
            "body_preview": body[:200].replace("\n", " ") if body else "",
            "topics_matched": "|".join(topics),
            "keywords_matched": "|".join(kws),
            "regions_matched": "|".join(regions),
            "index_date": it.get("index_date", ""),
            "is_relevant": "true" if topics else "false",
        })

    # Tulis CSV
    relevant = [r for r in rows if r["is_relevant"] == "true"]
    with (PROCESSED / "articles_relevant.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(sorted(relevant, key=lambda r: (r["source"], r["index_date"])))

    # Summary
    per_source = {}
    for r in relevant:
        s = r["source"]
        per_source.setdefault(s, {"total": 0, "with_body": 0})
        per_source[s]["total"] += 1
        per_source[s]["with_body"] += 1 if int(r["body_length"]) > 0 else 0

    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS / "normalize_news.json", {
        "step": "normalize_news",
        "total_relevant": len(relevant),
        "per_source": per_source,
        "output": "data/processed/news/articles_relevant.csv",
        "generated_at_utc": now_utc(),
    })
    for s, info in per_source.items():
        logging.info("%s: %d relevan, %d dengan body", s, info["total"], info["with_body"])
    logging.info("NORMALIZE SELESAI: %d artikel relevan -> articles_relevant.csv", len(relevant))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Collector News ARIF-Net.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Tahap 1: Scrape halaman indeks")
    p.add_argument("--source", default="all")
    p.add_argument("--dates", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--max-minutes", type=float, default=None)

    p = sub.add_parser("articles", help="Tahap 2: Scrape isi berita relevan")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)

    sub.add_parser("normalize", help="Tahap 3: Normalisasi -> CSV")

    args = ap.parse_args()
    if args.cmd == "index":
        return cmd_index(args)
    elif args.cmd == "articles":
        return cmd_articles(args)
    else:
        return cmd_normalize(args)


if __name__ == "__main__":
    sys.exit(main())
