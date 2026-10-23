"""GET /api/trade/{iso2} — top export partners with goods categories.

Served from the WITS ingestor's weekly top_export_partners record; the static
trade_flows.json is the fallback for countries WITS has no record for (those
that don't report trade, or before the first WITS run).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from exo.store.feature_store import FeatureStore

router = APIRouter()
_store = FeatureStore()

_DATA_PATH = Path(__file__).parent.parent / "data" / "trade_flows.json"
_TRADE_DATA: dict | None = None


def _load() -> dict:
    global _TRADE_DATA
    if _TRADE_DATA is None:
        _TRADE_DATA = json.loads(_DATA_PATH.read_text())
    return _TRADE_DATA


class TradePartner(BaseModel):
    iso2: str
    name: str
    trade_usd_b: float
    share_pct: float
    goods: list[str]


class TradeFlowResponse(BaseModel):
    country: str
    partners: list[TradePartner]
    source: str = "static"      # "wits" (live) or "static" (trade_flows.json)
    year: int | None = None     # data year, for live data


@router.get("/trade/{iso2}", response_model=TradeFlowResponse)
def get_trade_flows(iso2: str):
    """Top 5 export partners and goods categories for a country."""
    iso2 = iso2.upper()
    live = _store.get_latest(entity=iso2, signal_type="top_export_partners", source="wits")
    if live and live.metadata.get("partners"):
        return TradeFlowResponse(
            country=iso2,
            partners=[TradePartner(**p) for p in live.metadata["partners"]],
            source="wits",
            year=live.metadata.get("year"),
        )
    partners = _load().get(iso2)
    if not partners:
        raise HTTPException(status_code=404, detail=f"No trade data for '{iso2}'")
    return TradeFlowResponse(
        country=iso2,
        partners=[TradePartner(**p) for p in partners],
    )
