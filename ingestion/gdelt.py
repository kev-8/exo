"""GDELT ingestor — runs every 15 minutes via APScheduler.

Uses the ``gdeltdoc`` library to query GDELT 2.0 doc API.
Emits ``FeatureRecord`` with signal_type='news_sentiment'.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

_GDELT_AVAILABLE = False
try:
    from gdeltdoc import GdeltDoc, Filters

    _GDELT_AVAILABLE = True
except ImportError:
    logger.warning("gdeltdoc not installed; GDELTIngestor will return no data")


# Maps ISO country code → GDELT search keyword
COUNTRY_KEYWORD_MAP: dict[str, str] = {
    "US": "United States",
    "RU": "Russia",
    "CN": "China",
    "UA": "Ukraine",
    "IL": "Israel",
    "IR": "Iranian",
    "KP": "North Korea",
    "TW": "Taiwan",
    "IN": "India",
    "PK": "Pakistan",
}

# Simple positive / negative word lists for naive sentiment scoring
_POS = {"peace", "agreement", "ceasefire", "deal", "election", "growth", "recovery"}
_NEG = {"war", "attack", "conflict", "crisis", "sanction", "threat", "violence", "bomb"}


def _naive_sentiment(text: str) -> float:
    """Return sentiment in [-1, 1] based on word overlap."""
    words = set(text.lower().split())
    pos = len(words & _POS)
    neg = len(words & _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _tone_to_float(tone: str | float | None) -> float:
    """Convert GDELT tone string to float in [-1, 1]."""
    if tone is None:
        return 0.0
    try:
        raw = float(str(tone).split(",")[0])
        # GDELT tone ranges roughly -10 to +10; normalise to [-1, 1]
        return max(-1.0, min(1.0, raw / 10.0))
    except (ValueError, IndexError):
        return 0.0


class GDELTIngestor(BaseIngestor):
    """Ingest GDELT news events and emit sentiment + magnitude features."""

    source = "gdelt"

    def __init__(self, countries: dict[str, str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.countries = countries or COUNTRY_KEYWORD_MAP
        self._client = GdeltDoc() if _GDELT_AVAILABLE else None

    async def fetch(self) -> list[RawRecord]:
        if not _GDELT_AVAILABLE or self._client is None:
            logger.debug("gdeltdoc unavailable; skipping fetch")
            return []

        raws: list[RawRecord] = []
        now = self.utcnow()

        loop = asyncio.get_event_loop()

        for iso_code, keyword in self.countries.items():
            end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
            f = Filters(
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
                num_records=10,
            )

            articles = None
            for attempt in range(3):
                try:
                    articles = await loop.run_in_executor(
                        None, self._client.article_search, f
                    )
                    break
                except Exception as exc:
                    logger.warning("GDELT fetch attempt %d failed for %s: %s", attempt + 1, iso_code, exc)
                    await asyncio.sleep(5 * (attempt + 1))

            if articles is None or (hasattr(articles, "empty") and articles.empty):
                await asyncio.sleep(3)
                continue

            for _, row in articles.iterrows():
                raw = {
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "seendate": row.get("seendate", ""),
                    "tone": row.get("tone", 0),
                    "language": row.get("language", ""),
                    "domain": row.get("domain", ""),
                }
                raws.append(
                    RawRecord(
                        source=self.source,
                        entity=iso_code,
                        raw=raw,
                        fetched_at=now,
                    )
                )

            await asyncio.sleep(3)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = self.utcnow()
        title = raw.raw.get("title", "")
        tone = _tone_to_float(raw.raw.get("tone"))
        magnitude = abs(tone)

        # Attempt to parse seendate
        seen_str = raw.raw.get("seendate", "")
        try:
            as_of = datetime.strptime(seen_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            as_of = now

        sentiment_record = FeatureRecord(
            source=self.source,
            entity=raw.entity,
            signal_type="news_sentiment",
            value=tone,
            metadata={
                "title": title,
                "url": raw.raw.get("url", ""),
                "domain": raw.raw.get("domain", ""),
                "language": raw.raw.get("language", ""),
            },
            as_of_ts=as_of,
        )

        magnitude_record = FeatureRecord(
            source=self.source,
            entity=raw.entity,
            signal_type="news_magnitude",
            value=magnitude,
            metadata={"url": raw.raw.get("url", "")},
            as_of_ts=as_of,
        )

        return [sentiment_record, magnitude_record]
