"""GET /api/trade/{iso2} — top export partners with goods categories."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_DATA_PATH = Path(__file__).parent.parent / "data" / "trade_flows.json"
_TRADE_DATA: dict = json.loads(_DATA_PATH.read_text())


class TradePartner(BaseModel):
    iso2: str
    name: str
    trade_usd_b: float
    share_pct: float
    goods: list[str]


class TradeFlowResponse(BaseModel):
    country: str
    partners: list[TradePartner]


@router.get("/trade/{iso2}", response_model=TradeFlowResponse)
def get_trade_flows(iso2: str):
    """Top 5 export partners and goods categories for a country."""
    iso2 = iso2.upper()
    partners = _TRADE_DATA.get(iso2)
    if not partners:
        raise HTTPException(status_code=404, detail=f"No trade data for '{iso2}'")
    return TradeFlowResponse(
        country=iso2,
        partners=[TradePartner(**p) for p in partners],
    )
