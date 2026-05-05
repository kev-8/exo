"""GET /api/risk/{iso2} and /api/risk/{iso2}/history — risk index data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from exo.risk_index.engine import COUNTRIES
from exo.store.index_store import IndexStore

router = APIRouter()
_index_store = IndexStore()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class TierScoreResponse(BaseModel):
    score: float
    contributing_signals: list[str]


class DimensionResponse(BaseModel):
    name: str
    score: float
    weight: float
    contributing_signals: list[str]
    tiers: dict[str, TierScoreResponse]


class RiskSnapshotResponse(BaseModel):
    country: str
    composite_score: float
    structural_score: float
    short_term_score: float
    acute_score: float
    dimensions: list[DimensionResponse]
    as_of_ts: str
    computed_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise(snap) -> RiskSnapshotResponse:
    dims = [
        DimensionResponse(
            name=d.name,
            score=round(d.score, 4),
            weight=d.weight,
            contributing_signals=d.contributing_signals,
            tiers={
                tier: TierScoreResponse(
                    score=round(ts.score, 4),
                    contributing_signals=ts.contributing_signals,
                )
                for tier, ts in d.tier_scores.items()
            },
        )
        for d in snap.dimensions
    ]
    return RiskSnapshotResponse(
        country=snap.country,
        composite_score=round(snap.composite_score, 4),
        structural_score=round(snap.structural_score, 4),
        short_term_score=round(snap.short_term_score, 4),
        acute_score=round(snap.acute_score, 4),
        dimensions=dims,
        as_of_ts=snap.as_of_ts.isoformat(),
        computed_at=snap.computed_at.isoformat(),
    )


def _validate_iso2(iso2: str) -> str:
    iso2 = iso2.upper()
    if iso2 not in COUNTRIES:
        raise HTTPException(status_code=404, detail=f"Country '{iso2}' is not tracked")
    return iso2


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/risk/{iso2}", response_model=RiskSnapshotResponse)
def get_latest_risk(iso2: str):
    """Latest risk index snapshot for a country."""
    iso2 = _validate_iso2(iso2)
    snap = _index_store.get_latest(iso2)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail=f"No risk data available yet for '{iso2}' — pipeline may not have run",
        )
    return _serialise(snap)


@router.get("/risk/{iso2}/history", response_model=list[RiskSnapshotResponse])
def get_risk_history(
    iso2: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Historical risk index snapshots for a country (up to *days* back)."""
    iso2 = _validate_iso2(iso2)
    from datetime import timedelta
    start_ts = datetime.now(timezone.utc) - timedelta(days=days)
    snaps = _index_store.get_history(iso2, start_ts=start_ts, limit=days * 4)
    return [_serialise(s) for s in snaps]
