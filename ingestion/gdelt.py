"""GDELT ingestor — runs every 15 minutes via APScheduler.

Uses the ``gdeltdoc`` library to query GDELT 2.0 doc API.
Emits ``FeatureRecord`` with signal_type='news_sentiment' (timeline tone)
and 'news_magnitude' (normalised article count) per country.
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

# Maximum article count used to normalise magnitude to [0, 1]
MAX_ARTICLE_COUNT = 50


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
                num_records=50,
            )

            # --- Tone via timelinetone ---
            tone_score: float | None = None
            for attempt in range(3):
                try:
                    tone_df = await loop.run_in_executor(
                        None, self._client.timeline_search, "timelinetone", f
                    )
                    if tone_df is not None and not tone_df.empty:
                        # Series column is typically "Tone"; fall back to first numeric column
                        tone_col = "Tone" if "Tone" in tone_df.columns else next(
                            (c for c in tone_df.columns if c != "datetime"), None
                        )
                        if tone_col:
                            raw_tone = float(tone_df[tone_col].mean())
                            # GDELT tone roughly -10 to +10; normalise to [-1, 1]
                            tone_score = max(-1.0, min(1.0, raw_tone / 10.0))
                    break
                except Exception as exc:
                    logger.warning(
                        "GDELT timeline fetch attempt %d failed for %s: %s",
                        attempt + 1, iso_code, exc,
                    )
                    await asyncio.sleep(5 * (attempt + 1))

            await asyncio.sleep(5)

            # --- Article count via article_search ---
            article_count: int = 0
            for attempt in range(3):
                try:
                    articles = await loop.run_in_executor(
                        None, self._client.article_search, f
                    )
                    if articles is not None and not (hasattr(articles, "empty") and articles.empty):
                        article_count = len(articles)
                    break
                except Exception as exc:
                    logger.warning(
                        "GDELT article fetch attempt %d failed for %s: %s",
                        attempt + 1, iso_code, exc,
                    )
                    await asyncio.sleep(5 * (attempt + 1))

            if tone_score is None and article_count == 0:
                logger.debug("No GDELT data for %s", iso_code)
                await asyncio.sleep(3)
                continue

            raws.append(
                RawRecord(
                    source=self.source,
                    entity=iso_code,
                    raw={
                        "tone_score": tone_score,
                        "article_count": article_count,
                        "keyword": keyword,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    fetched_at=now,
                )
            )

            await asyncio.sleep(3)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw

        records: list[FeatureRecord] = []

        tone_score = d.get("tone_score")
        article_count = d.get("article_count", 0)

        if tone_score is not None:
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type="news_sentiment",
                    value=float(tone_score),
                    metadata={"keyword": d.get("keyword"), "window": f"{d.get('start_date')}:{d.get('end_date')}"},
                    as_of_ts=now,
                )
            )

        # Magnitude: normalised article count
        magnitude = min(1.0, article_count / MAX_ARTICLE_COUNT)
        records.append(
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="news_magnitude",
                value=magnitude,
                metadata={"article_count": article_count},
                as_of_ts=now,
            )
        )

        return records
