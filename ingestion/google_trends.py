"""Google Trends ingestor — relative search volume, every 2 hours."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

_PYTRENDS_AVAILABLE = False
try:
    from pytrends.request import TrendReq

    _PYTRENDS_AVAILABLE = True
except ImportError:
    logger.warning("pytrends not installed; GoogleTrendsIngestor will return no data")

# Keywords tracked (grouped — Google Trends allows max 5 per request)
KEYWORD_GROUPS: list[list[str]] = [
    ["war", "conflict", "sanctions", "ceasefire", "invasion"],
    ["election", "vote", "polling", "approval", "incumbent"],
    ["recession", "inflation", "unemployment", "GDP", "federal reserve"],
]


class GoogleTrendsIngestor(BaseIngestor):
    """Ingest Google Trends relative search interest."""

    source = "google_trends"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pytrends = TrendReq(hl="en-US", tz=0) if _PYTRENDS_AVAILABLE else None

    async def fetch(self) -> list[RawRecord]:
        if not _PYTRENDS_AVAILABLE or self._pytrends is None:
            logger.warning("pytrends unavailable; skipping fetch")
            return []

        raws: list[RawRecord] = []
        now = self.utcnow()

        for group in KEYWORD_GROUPS:
            try:
                self._pytrends.build_payload(group, timeframe="now 1-d", geo="")
                df = self._pytrends.interest_over_time()
                if df is None or df.empty:
                    continue
                last_row = df.iloc[-1]
                values = {kw: float(last_row.get(kw, 0)) for kw in group if kw in last_row}
                if values:
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity="_".join(group[:2]),
                            raw=values,
                            fetched_at=now,
                        )
                    )
            except Exception as exc:
                logger.warning("Google Trends fetch failed for group %s: %s", group, exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        records: list[FeatureRecord] = []

        for keyword, value in raw.raw.items():
            normalised = float(value) / 100.0  # Trends values are 0-100
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=keyword,
                    signal_type="search_volume",
                    value=normalised,
                    metadata={"raw_value": float(value)},
                    as_of_ts=now,
                )
            )

        return records
