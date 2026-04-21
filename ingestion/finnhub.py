"""Finnhub ingestor — news sentiment via FinBERT, every 30 minutes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"
CATEGORIES = ["general", "forex", "merger"]

_pipeline = None


def _get_pipeline():
    """Lazy-load FinBERT pipeline (downloads model on first call)."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info("Loading FinBERT sentiment model...")
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT model loaded")
    return _pipeline


def _score_texts(texts: list[str]) -> float:
    """Run FinBERT on a list of texts, return mean sentiment in [-1, 1]."""
    if not texts:
        return 0.0
    pipe = _get_pipeline()
    results = pipe(texts, batch_size=16)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = [label_map.get(r["label"].lower(), 0.0) * r["score"] for r in results]
    return sum(scores) / len(scores)


class FinnhubIngestor(BaseIngestor):
    """Ingest Finnhub news and score sentiment locally with FinBERT."""

    source = "finnhub"

    async def fetch(self) -> list[RawRecord]:
        if not config.FINNHUB_API_KEY:
            logger.warning("FINNHUB_API_KEY not configured; skipping fetch")
            return []

        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for category in CATEGORIES:
                try:
                    resp = await client.get(
                        f"{FINNHUB_BASE}/news",
                        params={"category": category, "token": config.FINNHUB_API_KEY},
                    )
                    resp.raise_for_status()
                    articles = resp.json()
                    if not articles:
                        continue

                    texts = [
                        f"{a.get('headline', '')} {a.get('summary', '')}".strip()
                        for a in articles[:20]
                        if a.get("headline") or a.get("summary")
                    ]

                    if not texts:
                        continue

                    mean_sentiment = _score_texts(texts)
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity=category,
                            raw={
                                "category": category,
                                "mean_sentiment": mean_sentiment,
                                "n_articles": len(texts),
                                "headlines": [a.get("headline", "")[:100] for a in articles[:3]],
                            },
                            fetched_at=now,
                        )
                    )
                except Exception as exc:
                    logger.warning("Finnhub fetch failed for %s: %s", category, exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        sentiment = float(raw.raw.get("mean_sentiment", 0.0))
        clamped = max(-1.0, min(1.0, sentiment))
        # Normalise to [0, 1]: negative sentiment (high stress) → score near 1.0
        risk_score = (1.0 - clamped) / 2.0

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="news_sentiment",
                value=risk_score,
                metadata={
                    "category": raw.raw.get("category"),
                    "n_articles": raw.raw.get("n_articles"),
                    "sample_headlines": raw.raw.get("headlines", []),
                    "raw_sentiment": clamped,
                },
                as_of_ts=now,
            )
        ]
