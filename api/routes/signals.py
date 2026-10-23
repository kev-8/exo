"""GET /api/signals/{iso2} — recent feature records for the signal feed."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from exo.store.feature_store import FeatureStore

router = APIRouter()
_store = FeatureStore()

# Country-level signals shown in the feed, by source. Read with get_latest
# (Redis first, then newest partition), so every signal appears once with its
# latest value. Sources keyed by something other than a country (google_trends
# topics, eia benchmarks, kalshi tickers, finnhub categories) aren't listed.
FEED_SIGNALS: dict[str, tuple[str, ...]] = {
    "gdelt": ("news_sentiment", "news_magnitude"),
    "world_bank": (
        "gdp_growth", "inflation_rate", "unemployment_rate", "debt_to_gdp", "gross_savings",
        "energy_imports_pct", "trade_openness", "exports_pct_gdp", "imports_pct_gdp",
        "electricity_access",
    ),
    "fred": (   # US only
        "economic_indicator", "fred_unrate", "fred_cpiaucsl", "fred_fedfunds",
        "fred_t10y2y", "fred_umcsent",
    ),
    "wgi": ("political_stability", "voice_accountability", "rule_of_law", "control_of_corruption"),
    "unga_votes": ("ideal_point_variance",),
    "ofac": ("sdn_entity_count",),
    "wits": ("trade_concentration", "secondary_exposure"),
    "ucdp_ged": ("ucdp_ged_events", "ucdp_ged_fatalities"),
    "ucdp_candidate": ("ucdp_candidate_events",),
    "comtrade": ("hw_contested_share", "supplier_chokepoint", "cn_mineral_dependence"),
    "export_controls": ("entity_list_count", "jp_eul_count", "cn_ec_count"),
    "digital_infra": ("cables_in_service", "ixp_count", "facility_count"),
}


class SignalResponse(BaseModel):
    source: str
    signal_type: str
    value: float
    as_of_ts: str
    metadata: dict


@router.get("/signals/{iso2}", response_model=list[SignalResponse])
def get_latest_signals(
    iso2: str,
    limit: int = Query(default=40, ge=1, le=60),
):
    """Latest value of each country-level signal for *iso2*, newest first."""
    iso2 = iso2.upper()
    records = []
    for source, signal_types in FEED_SIGNALS.items():
        if source == "fred" and iso2 != "US":
            continue
        for signal_type in signal_types:
            rec = _store.get_latest(entity=iso2, signal_type=signal_type, source=source)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.as_of_ts, reverse=True)
    return [
        SignalResponse(
            source=r.source,
            signal_type=r.signal_type,
            value=round(r.value, 4),
            as_of_ts=r.as_of_ts.isoformat(),
            metadata=r.metadata,
        )
        for r in records[:limit]
    ]
