"""GET /api/signals/{iso2} — recent feature records for the signal feed."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from exo.models import FeatureQuery
from exo.store.feature_store import FeatureStore

router = APIRouter()
_store = FeatureStore()

# Sources that store records keyed by ISO2 entity (e.g. entity="RU").
# Excludes google_trends (topic strings), eia (HH/WTI), kalshi (tickers),
# and finnhub (category names) — those never match an ISO2 entity filter
# and loading their partitions for every request is wasteful.
_ISO2_SOURCES = ["gdelt", "world_bank", "fred"]


class SignalResponse(BaseModel):
    source: str
    signal_type: str
    value: float
    as_of_ts: str
    metadata: dict


@router.get("/signals/{iso2}", response_model=list[SignalResponse])
def get_latest_signals(
    iso2: str,
    limit: int = Query(default=10, ge=1, le=50),
):
    """Most recent feature records for *iso2* across all country-level sources."""
    iso2 = iso2.upper()

    # Query each source individually so we hit a specific partition glob
    # (source=gdelt/date=*/...) rather than the expensive source=* scan.
    all_records = []
    for source in _ISO2_SOURCES:
        # fred only carries US data; skip for other countries
        if source == "fred" and iso2 != "US":
            continue
        records = _store.read(FeatureQuery(entity=iso2, source=source, limit=limit))
        all_records.extend(records)

    # Sort by as_of_ts descending, deduplicate by (source, signal_type), cap at limit
    seen: set[tuple[str, str]] = set()
    results = []
    for r in sorted(all_records, key=lambda x: x.as_of_ts, reverse=True):
        key = (r.source, r.signal_type)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            SignalResponse(
                source=r.source,
                signal_type=r.signal_type,
                value=round(r.value, 4),
                as_of_ts=r.as_of_ts.isoformat(),
                metadata=r.metadata,
            )
        )
        if len(results) >= limit:
            break

    return results
