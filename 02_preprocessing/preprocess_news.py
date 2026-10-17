"""
ARIF-Net -- Preprocessing berita: sentimen & agregasi fitur news.

Membaca articles_relevant.csv + body text -> menghasilkan fitur news per hari:
  - news_count: jumlah berita relevan per hari per komoditas
  - sentiment_mean/min/max: skor sentimen rata-rata/min/max per hari
  - topic_counts: jumlah berita per topik per hari
  - body_avg_length: panjang rata-rata isi berita

Model sentimen: Indonesian sentiment dari HuggingFace (indobenchmark/indobert-base-p1)
  atau fallback ke VADER/TextBlob kalau GPU tidak tersedia.

Output: data/processed/news/news_features.csv

Contoh:
  python -m 02_preprocessing.preprocess_news [--no-sentiment]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_PROCESSED, DATA_RAW, load_project_config, now_utc, save_json, load_json

NEWS_PROCESSED = DATA_PROCESSED / "news"
OUT = DATA_PROCESSED / "news"


def load_articles() -> pd.DataFrame:
    """Baca articles_relevant.csv dari collection/normalize."""
    path = NEWS_PROCESSED / "articles_relevant.csv"
    if not path.exists():
        logging.warning("articles_relevant.csv tidak ditemukan!")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["published_at_wib"])
    return df


def simple_sentiment(text: str) -> float:
    """Skor sentimen sederhana berbasis kata kunci (fallback jika tidak ada model NLP).

    Positif: stabilisasi, turun, murah, surplus, panen raya, aman, normal
    Negatif: naik, mahal, gagal panen, krisis, langka, defisit, banjir, kekeringan
    Return: skor antara -1.0 dan 1.0
    """
    if not text:
        return 0.0
    text = text.lower()
    positive = ["stabil", "turun", "murah", "surplus", "panen raya", "aman",
                "normal", "melimpah", "terkendali", "terjangkau"]
    negative = ["naik", "mahal", "gagal panen", "krisis", "langka", "defisit",
                "banjir", "kekeringan", "kelangkaan", "melonjak", "meningkat tajam"]
    pos_count = sum(1 for w in positive if w in text)
    neg_count = sum(1 for w in negative if w in text)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_sentiment(articles: pd.DataFrame, use_model: bool = False) -> pd.DataFrame:
    """Hitung skor sentimen untuk setiap artikel."""
    text_dir = NEWS_PROCESSED / "article_texts"
    sentiments = []

    if use_model:
        try:
            from transformers import pipeline
            logging.info("Loading sentiment model (indobenchmark/indobert-base-p1)...")
            classifier = pipeline("sentiment-analysis",
                                  model="mdhugol/indonesia-bert-sentiment-classification",
                                  truncation=True, max_length=512)
            for _, row in articles.iterrows():
                body_file = text_dir / f"{row['article_id']}.txt"
                text = body_file.read_text(encoding="utf-8")[:500] if body_file.exists() else row.get("title", "")
                try:
                    result = classifier(text)[0]
                    score = result["score"] if result["label"] == "positive" else -result["score"]
                    sentiments.append(score)
                except Exception:
                    sentiments.append(simple_sentiment(text))
            articles = articles.copy()
            articles["sentiment"] = sentiments
            return articles
        except ImportError:
            logging.warning("transformers tidak tersedia, menggunakan keyword-based sentiment.")

    # Fallback: keyword-based
    for _, row in articles.iterrows():
        body_file = text_dir / f"{row['article_id']}.txt"
        text = ""
        if body_file.exists():
            text = body_file.read_text(encoding="utf-8")
        full_text = str(row.get("title", "")) + " " + text
        sentiments.append(simple_sentiment(full_text))

    articles = articles.copy()
    articles["sentiment"] = sentiments
    return articles


def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    """Agregasi berita per hari -> fitur tabular."""
    if articles.empty:
        return pd.DataFrame()

    # Ekstrak tanggal dari published_at_wib
    articles["pub_date"] = pd.to_datetime(articles["published_at_wib"]).dt.date

    # Parse topics
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "news_scope.json"
    scope = load_json(cfg_path)
    all_topics = list(scope["topics"].keys())

    daily_records = []
    for pub_date, group in articles.groupby("pub_date"):
        record = {
            "date": str(pub_date),
            "news_count": len(group),
            "news_count_with_body": int((group["body_length"].astype(int) > 0).sum()),
            "sentiment_mean": float(group["sentiment"].mean()),
            "sentiment_min": float(group["sentiment"].min()),
            "sentiment_max": float(group["sentiment"].max()),
            "sentiment_std": float(group["sentiment"].std()) if len(group) > 1 else 0.0,
            "body_avg_length": float(group["body_length"].astype(int).mean()),
        }
        # Count per topic
        for topic in all_topics:
            count = sum(1 for _, r in group.iterrows()
                        if topic in str(r.get("topics_matched", "")))
            record[f"news_topic_{topic}"] = count
        daily_records.append(record)

    return pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocessing news ARIF-Net.")
    ap.add_argument("--no-sentiment", action="store_true", help="Skip sentiment model, gunakan keyword-based")
    ap.add_argument("--use-model", action="store_true", help="Gunakan model NLP untuk sentimen")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Loading articles...")
    articles = load_articles()

    if articles.empty:
        logging.warning("Tidak ada artikel, membuat file kosong.")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT / "news_features.csv", index=False)
        return 0

    logging.info("Articles loaded: %d baris", len(articles))
    logging.info("Computing sentiment...")
    articles = compute_sentiment(articles, use_model=args.use_model and not args.no_sentiment)

    logging.info("Aggregating daily features...")
    daily = aggregate_daily(articles)

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "news_features.csv", index=False)
    logging.info("News features: %d hari -> news_features.csv", len(daily))

    # Save articles with sentiment
    articles.to_csv(OUT / "articles_with_sentiment.csv", index=False)

    save_json(OUT / "preprocess_news_report.json", {
        "step": "preprocess_news",
        "n_articles": len(articles),
        "n_days": len(daily),
        "sentiment_method": "model" if args.use_model else "keyword",
        "finished_at_utc": now_utc(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
