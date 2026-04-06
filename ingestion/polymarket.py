"""Polymarket ingestor — active market prices via Gamma API, hourly.

Fetches events filtered by geopolitically relevant tags. Applies two
selection strategies per event:
  1. Longest-dated market with liquidity > $500 (structural signal)
  2. Any near-term market (<30 days) with liquidity > $5,000 (acute signal)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
TARGET_TAGS = ["politics", "world", "economy", "elections", "tech"]

MIN_LIQUIDITY = 500       # USD — baseline filter
NEAR_TERM_DAYS = 30       # days — threshold for near-term markets
NEAR_TERM_LIQUIDITY = 5000  # USD — higher bar for near-term markets


def _liquidity(m: dict) -> float:
    """Return liquidity as float, trying liquidityNum then liquidity field."""
    val = m.get("liquidityNum") or m.get("liquidity")
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _select_markets(markets: list[dict], now: datetime) -> list[dict]:
    """Select structural (longest-dated) and near-term (high-liquidity) markets."""
    selected: dict[str, dict] = {}  # market_id → market

    # Strategy 1: longest-dated market per event with liquidity > $500
    liquid = [m for m in markets if _liquidity(m) > MIN_LIQUIDITY and m.get("endDateIso")]
    if liquid:
        longest = max(liquid, key=lambda m: m["endDateIso"])
        selected[str(longest.get("id", ""))] = longest

    # Strategy 2: near-term markets (<30 days) with liquidity > $5,000
    cutoff = (now + timedelta(days=NEAR_TERM_DAYS)).isoformat()
    near_term = [
        m for m in markets
        if m.get("endDateIso")
        and m["endDateIso"] <= cutoff
        and _liquidity(m) > NEAR_TERM_LIQUIDITY
    ]
    for m in near_term:
        selected[str(m.get("id", ""))] = m

    return list(selected.values())


class PolymarketIngestor(BaseIngestor):
    """Ingest Polymarket market prices filtered by geopolitical tags."""

    source = "polymarket"

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()
        seen_ids: set[str] = set()

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                for tag in TARGET_TAGS:
                    try:
                        resp = await client.get(
                            f"{GAMMA_API}/events",
                            params={
                                "active": "true",
                                "closed": "false",
                                "limit": 100,
                                "tag_slug": tag,
                            },
                        )
                        resp.raise_for_status()
                        events = resp.json()

                        for event in events:
                            markets = event.get("markets", [])
                            selected = _select_markets(markets, now)

                            for market in selected:
                                market_id = str(market.get("id", ""))
                                if not market_id or market_id in seen_ids:
                                    continue
                                seen_ids.add(market_id)
                                raws.append(
                                    RawRecord(
                                        source=self.source,
                                        entity=market_id,
                                        raw={
                                            **market,
                                            "tag": tag,
                                            "event_title": event.get("title", ""),
                                        },
                                        fetched_at=now,
                                    )
                                )
                    except Exception as exc:
                        logger.warning("Polymarket fetch failed for tag %s: %s", tag, exc)

        except Exception as exc:
            logger.error("Polymarket fetch failed: %s", exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw

        outcome_prices = d.get("outcomePrices")
        if not outcome_prices:
            return []

        try:
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            prices = [float(p) for p in outcome_prices]
            price = prices[0]  # first outcome (typically "Yes") probability
        except (ValueError, TypeError, IndexError):
            return []

        # Flag near-term markets for downstream use
        is_near_term = False
        end_date = d.get("endDateIso", "")
        if end_date:
            cutoff = (now + timedelta(days=NEAR_TERM_DAYS)).isoformat()
            is_near_term = end_date <= cutoff

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="polymarket_price",
                value=price,
                metadata={
                    "question": str(d.get("question", ""))[:200],
                    "event_title": str(d.get("event_title", ""))[:200],
                    "tag": d.get("tag"),
                    "volume": d.get("volumeNum") or d.get("volume"),
                    "liquidity": _liquidity(d),
                    "end_date": end_date,
                    "near_term": is_near_term,
                },
                as_of_ts=now,
            )
        ]
