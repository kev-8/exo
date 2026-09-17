"""GDELT ingestor — runs every 15 minutes via APScheduler.

Uses the ``gdeltdoc`` library to query GDELT 2.0 doc API.
Emits ``FeatureRecord`` with signal_type='news_sentiment' (timeline tone) per country.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

# GDELT throttles at one request per 5 seconds ("Please limit requests to one
# every 5 seconds", HTTP 429). Keep a margin above the documented floor; 
# 20 countries * 7s is ~140s per cycle, comfortable inside the 15-minute schedule.
_REQUEST_INTERVAL_SEC = 7.0

_GDELT_AVAILABLE = False
try:
    from gdeltdoc import GdeltDoc, Filters

    _GDELT_AVAILABLE = True
except ImportError:
    logger.warning("gdeltdoc not installed; GDELTIngestor will return no data")


def _describe(exc: BaseException) -> str:
    """Render an exception usefully even when it carries no message.
        Always surface the type, and the HTTP status/body
        when the exception carries a response.
    """
    parts = [type(exc).__name__]
    text = str(exc).strip()
    if text:
        parts.append(text)
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            parts.append(f"HTTP {status}")
        body = (getattr(response, "text", "") or "").strip()
        if body:
            parts.append(body[:200])
    return ": ".join(parts)


# Maps ISO country code → GDELT search keyword
COUNTRY_KEYWORD_MAP: dict[str, str] = {
    "US": "United States", "RU": "Russia",        "CN": "China",
    "UA": "Ukraine",       "IL": "Israel",         "IR": "Iranian",
    "KP": "North Korea",   "TW": "Taiwan",         "IN": "India",
    "PK": "Pakistan",      "HT": "Haiti",          "BR": "Brazil",
    "MX": "Mexico",        "NG": "Nigeria",         "KE": "Kenya",
    "ZA": "South Africa",  "FR": "France",          "GB": "United Kingdom",
    "MY": "Malaysia",      "CL": "Chile",
}


class GDELTIngestor(BaseIngestor):
    """Ingest GDELT news tone via timelinetone and emit news_sentiment features."""

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

        throttled = 0
        for i, (iso_code, keyword) in enumerate(self.countries.items()):
            # Pace *between* requests rather than after each one, so the gap
            # is enforced even when a query fails early.
            if i:
                await asyncio.sleep(_REQUEST_INTERVAL_SEC)

            end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
            f = Filters(
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
            )

            tone_score: float | None = None
            try:
                tone_df = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._client.timeline_search, "timelinetone", f
                    ),
                    timeout=20.0,
                )
                if tone_df is not None and not tone_df.empty:
                    tone_col = "Tone" if "Tone" in tone_df.columns else next(
                        (c for c in tone_df.columns if c != "datetime"), None
                    )
                    if tone_col:
                        raw_tone = float(tone_df[tone_col].mean())
                        tone_score = max(-1.0, min(1.0, raw_tone / 10.0))
            except asyncio.TimeoutError:
                logger.warning("GDELT fetch timed out for %s", iso_code)
            except Exception as exc:
                if type(exc).__name__ == "RateLimitError":
                    throttled += 1
                logger.warning("GDELT fetch failed for %s: %s", iso_code, _describe(exc))

            if tone_score is None:
                logger.debug("No GDELT tone data for %s", iso_code)
                continue

            raws.append(
                RawRecord(
                    source=self.source,
                    entity=iso_code,
                    raw={
                        "tone_score": tone_score,
                        "keyword": keyword,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    fetched_at=now,
                )
            )

        if throttled:
            logger.error(
                "GDELT rate-limited on %d/%d countries this cycle — requests are "
                "outpacing the 1-per-5s limit, or this IP is still in a throttle "
                "penalty window from earlier over-limit traffic",
                throttled, len(self.countries),
            )
        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw

        tone_score = d.get("tone_score")
        if tone_score is None:
            return []

        # Normalise to [0, 1]: negative tone (high risk) → score near 1.0
        risk_score = max(0.0, min(1.0, (1.0 - float(tone_score)) / 2.0))

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="news_sentiment",
                value=risk_score,
                metadata={"keyword": d.get("keyword"), "window": f"{d.get('start_date')}:{d.get('end_date')}", "raw_tone": float(tone_score)},
                as_of_ts=now,
            )
        ]
