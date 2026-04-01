"""Finnhub ingestor — news sentiment API, every 30 minutes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Market news categories to track
CATEGORIES = ["general", "forex", "crypto", "merger"]


class FinnhubIngestor(BaseIngestor):
    """Ingest Finnhub news sentiment signals."""

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

                    # Aggregate sentiment across articles
                    sentiments: list[float] = []
                    for art in articles[:20]:
                        sentiment = art.get("sentiment")
                        if sentiment is not None:
                            try:
                                sentiments.append(float(sentiment))
                            except (ValueError, TypeError):
                                pass

                    if sentiments:
                        mean_sent = sum(sentiments) / len(sentiments)
                        raws.append(
                            RawRecord(
                                source=self.source,
                                entity=category,
                                raw={
                                    "category": category,
                                    "mean_sentiment": mean_sent,
                                    "n_articles": len(sentiments),
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
        # Finnhub sentiment is typically in [-1, 1]
        normalised = max(-1.0, min(1.0, sentiment))

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="news_sentiment",
                value=normalised,
                metadata={
                    "category": raw.raw.get("category"),
                    "n_articles": raw.raw.get("n_articles"),
                    "sample_headlines": raw.raw.get("headlines", []),
                },
                as_of_ts=now,
            )
        ]
