"""UCDP conflict event ingestors — weekly.

Two ingestors sharing this module:

  UCDPGEDIngestor       — Georeferenced Events Dataset (GED), annual releases.
                          Feeds the *structural* tier of conflict_intensity.
  UCDPCandidateIngestor — Candidate dataset (current year, quarterly revisions).
                          Feeds the *short-term* tier of conflict_intensity.

API: https://ucdpapi.pcr.uu.se/api/  (no key required)

Normalisation: log-scale against a rolling max observed across all countries
and all time, read from the feature store at normalise() time.  This keeps
scores stable relative to the worst observed period rather than a fixed ceiling.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureQuery, FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

UCDP_BASE = "https://ucdpapi.pcr.uu.se/api"

# ISO 3166-1 alpha-2 → UCDP country_id mapping (GW codes used by UCDP)
# Source: https://ucdp.uu.se/downloads/
ISO2_TO_UCDP: dict[str, int] = {
    "US": 2,
    "RU": 365,
    "CN": 710,
    "UA": 369,
    "IL": 666,
    "IR": 630,
    "IN": 750,
    "PK": 770,
    "KP": 731,
    "TW": 713,
}

# Fallback rolling-max values used when the feature store has no history yet
_FALLBACK_MAX_EVENTS = 500.0
_FALLBACK_MAX_FATALITIES = 10_000.0


def _log_normalise(value: float, rolling_max: float, fallback: float) -> float:
    """log(1 + v) / log(1 + max), clamped to [0, 1]."""
    denom = math.log1p(max(rolling_max, fallback))
    return min(1.0, math.log1p(max(0.0, value)) / denom)


class _UCDPBase(BaseIngestor):
    """Shared fetch/normalise logic for GED and Candidate ingestors."""

    source: str                # overridden by subclass
    _endpoint: str             # overridden by subclass (e.g. "gedevents/23.1")
    _events_signal: str        # signal_type for event count
    _fatalities_signal: str    # signal_type for fatalities (GED only; "" = skip)

    async def _fetch_country(
        self,
        client: httpx.AsyncClient,
        country_id: int,
        iso2: str,
        now: datetime,
    ) -> RawRecord | None:
        """Fetch all events for a country from the UCDP paged API."""
        events: list[dict] = []
        page = 1
        page_size = 1000

        while True:
            try:
                resp = await client.get(
                    f"{UCDP_BASE}/{self._endpoint}",
                    params={
                        "country_id": country_id,
                        "pagesize": page_size,
                        "page": page,
                    },
                )
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                logger.debug("UCDP fetch failed country=%s page=%d: %s", iso2, page, exc)
                break

            result = body.get("Result", [])
            events.extend(result)

            total_count = body.get("TotalCount", 0)
            if len(events) >= total_count or not result:
                break
            page += 1

        if not events:
            return None

        total_fatalities = sum(
            int(e.get("best", 0) or 0) for e in events
        )
        return RawRecord(
            source=self.source,
            entity=iso2,
            raw={
                "event_count": len(events),
                "total_fatalities": total_fatalities,
            },
            fetched_at=now,
        )

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=60.0) as client:
            for iso2, country_id in ISO2_TO_UCDP.items():
                raw = await self._fetch_country(client, country_id, iso2, now)
                if raw is not None:
                    raws.append(raw)

        return raws

    def _rolling_max(self, signal_type: str) -> float:
        """Read the max observed value for *signal_type* across all history."""
        try:
            records = self.store.read(FeatureQuery(
                signal_type=signal_type,
                source=self.source,
                limit=10_000,
            ))
            if records:
                return max(r.value for r in records)
        except Exception as exc:
            logger.debug("Could not read rolling max for %s: %s", signal_type, exc)
        return 0.0

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw
        records: list[FeatureRecord] = []

        event_count = float(d.get("event_count", 0))
        total_fatalities = float(d.get("total_fatalities", 0))

        # Rolling max — read from store so normalisation adapts over time
        max_events = max(self._rolling_max(self._events_signal), event_count)
        events_score = _log_normalise(event_count, max_events, _FALLBACK_MAX_EVENTS)
        records.append(FeatureRecord(
            source=self.source,
            entity=raw.entity,
            signal_type=self._events_signal,
            value=round(events_score, 4),
            metadata={"raw_event_count": event_count, "rolling_max": max_events},
            as_of_ts=now,
        ))

        if self._fatalities_signal:
            max_fat = max(self._rolling_max(self._fatalities_signal), total_fatalities)
            fat_score = _log_normalise(total_fatalities, max_fat, _FALLBACK_MAX_FATALITIES)
            records.append(FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type=self._fatalities_signal,
                value=round(fat_score, 4),
                metadata={"raw_fatalities": total_fatalities, "rolling_max": max_fat},
                as_of_ts=now,
            ))

        return records


class UCDPGEDIngestor(_UCDPBase):
    """UCDP Georeferenced Events Dataset — structural conflict signal.

    Fetches the latest completed year of verified conflict events.
    Cadence: weekly (annual data releases; weekly polling picks up new releases).
    """

    source = "ucdp_ged"
    _endpoint = "gedevents/24.1"   # latest stable GED release; update annually
    _events_signal = "ucdp_ged_events"
    _fatalities_signal = "ucdp_ged_fatalities"


class UCDPCandidateIngestor(_UCDPBase):
    """UCDP Candidate dataset — short-term conflict signal.

    Fetches current/recent year data, subject to quarterly revision.
    Cadence: weekly (quarterly revision cycle).
    """

    source = "ucdp_candidate"
    _endpoint = "gedevents/candidates"
    _events_signal = "ucdp_candidate_events"
    _fatalities_signal = ""   # candidate dataset omits best-estimate fatalities
