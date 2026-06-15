"""FRED ingestor — macro economic indicators, runs every 12 hours.

Uses the ``fredapi`` library.  Emits a composite economic_indicator score.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

_FRED_AVAILABLE = False
try:
    from fredapi import Fred

    _FRED_AVAILABLE = True
except ImportError:
    logger.warning("fredapi not installed; FREDIngestor will return no data")

# Series to include in the composite score and their stress-direction sign
# sign=+1 means higher value → higher economic stress
# sign=-1 means higher value → lower stress (e.g. GDP growth)
SERIES: dict[str, int] = {
    "UNRATE": 1,      # Unemployment rate
    "CPIAUCSL": 1,    # CPI (inflation proxy)
    "FEDFUNDS": 1,    # Federal funds rate
    "T10Y2Y": -1,     # 10Y-2Y yield spread (inversion = stress)
    "UMCSENT": -1,    # Consumer sentiment
}


class FREDIngestor(BaseIngestor):
    """Ingest FRED macro series and emit economic_indicator features."""

    source = "fred"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fred = Fred(api_key=config.FRED_API_KEY) if _FRED_AVAILABLE and config.FRED_API_KEY else None

    async def fetch(self) -> list[RawRecord]:
        if not _FRED_AVAILABLE or self._fred is None:
            logger.warning("FRED client unavailable; skipping fetch")
            return []

        now = self.utcnow()
        loop = asyncio.get_event_loop()

        def _fetch_all() -> dict[str, float]:
            values: dict[str, float] = {}
            for series_id in SERIES:
                try:
                    s = self._fred.get_series(series_id, limit=1)
                    if s is not None and len(s) > 0:
                        values[series_id] = float(s.iloc[-1])
                except Exception as exc:
                    logger.warning("FRED series %s failed: %s", series_id, exc)
            return values

        try:
            values = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_all), timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.warning("FRED fetch timed out after 60s")
            return []

        if not values:
            return []
        return [RawRecord(source=self.source, entity="US", raw=values, fetched_at=now)]

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        values = raw.raw

        # Build individual records + composite
        records: list[FeatureRecord] = []
        stress_scores: list[float] = []

        for series_id, sign in SERIES.items():
            val = values.get(series_id)
            if val is None:
                continue
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type=f"fred_{series_id.lower()}",
                    value=val,
                    metadata={"series_id": series_id, "sign": sign},
                    as_of_ts=now,
                )
            )
            # Simple z-score normalisation placeholder: use raw value signed
            stress_scores.append(sign * val)

        if stress_scores:
            # Normalise composite to [0, 1] with sigmoid-like transform
            import math
            mean_stress = sum(stress_scores) / len(stress_scores)
            composite = 1 / (1 + math.exp(-mean_stress / 10))
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type="economic_indicator",
                    value=composite,
                    metadata={"component_series": list(values.keys())},
                    as_of_ts=now,
                )
            )

        return records
