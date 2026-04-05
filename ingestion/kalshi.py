"""Kalshi ingestor — RSA-PSS auth, WebSocket orderbook, REST metadata.

WebSocket maintains a persistent connection for live orderbook updates.
REST polls market metadata every 15 minutes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, KalshiMarket, RawRecord

logger = logging.getLogger(__name__)

_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    _CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not installed; Kalshi auth disabled")


def _load_private_key():
    """Load RSA private key from KALSHI_PRIVATE_KEY environment variable (PEM content)."""
    if not _CRYPTO_AVAILABLE:
        return None
    pem = config.KALSHI_PRIVATE_KEY
    if not pem:
        logger.warning("KALSHI_PRIVATE_KEY env var not set; Kalshi auth disabled")
        return None
    return serialization.load_pem_private_key(pem.encode(), password=None)


def _build_auth_headers(method: str, path: str) -> dict[str, str]:
    """Build RSA-PSS signed auth headers for Kalshi API."""
    if not _CRYPTO_AVAILABLE:
        return {}
    private_key = _load_private_key()
    if private_key is None:
        return {}

    ts_ms = str(int(time.time() * 1000))
    msg = ts_ms + method.upper() + path
    signature = private_key.sign(
        msg.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode()
    return {
        "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }


class KalshiIngestor(BaseIngestor):
    """Ingest Kalshi market data via REST + WebSocket."""

    source = "kalshi"

    def __init__(self, tickers: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tickers = tickers or []
        self._ws_task: asyncio.Task | None = None
        self._orderbook: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # REST fetch
    # ------------------------------------------------------------------

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch markets page
            try:
                path = "/markets"
                headers = _build_auth_headers("GET", "/trade-api/v2" + path)
                resp = await client.get(
                    config.KALSHI_BASE_URL + path,
                    headers=headers,
                    params={"status": "open", "limit": 200},
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Kalshi rate limited; retry after %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    return raws

                resp.raise_for_status()
                data = resp.json()
                markets = data.get("markets", [])

                for m in markets:
                    ob = self._orderbook.get(m.get("ticker", ""), {})
                    raw = {**m, "orderbook": ob}
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity=m.get("ticker", ""),
                            raw=raw,
                            fetched_at=now,
                        )
                    )
            except httpx.HTTPStatusError as exc:
                logger.error("Kalshi REST error: %s", exc)
            except Exception as exc:
                logger.error("Kalshi fetch failed: %s", exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        m = raw.raw
        ticker = m.get("ticker", raw.entity)
        now = raw.fetched_at

        yes_ask = float(m.get("yes_ask", 0) or 0) / 100
        yes_bid = float(m.get("yes_bid", 0) or 0) / 100
        no_ask = float(m.get("no_ask", 0) or 0) / 100
        no_bid = float(m.get("no_bid", 0) or 0) / 100
        volume = float(m.get("volume", 0) or 0)
        spread = round(yes_ask - yes_bid, 4)
        mid = round((yes_ask + yes_bid) / 2, 4) if yes_ask and yes_bid else yes_ask

        close_str = m.get("close_time") or m.get("expected_expiration_time")
        try:
            close_time = datetime.fromisoformat(str(close_str).rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            close_time = None

        records = [
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_price",
                value=mid,
                metadata={
                    "yes_ask": yes_ask,
                    "yes_bid": yes_bid,
                    "no_ask": no_ask,
                    "no_bid": no_bid,
                    "spread": spread,
                    "status": m.get("status", ""),
                    "title": m.get("title", ""),
                    "close_time": close_str,
                },
                ticker=ticker,
                as_of_ts=now,
            ),
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_volume",
                value=volume,
                metadata={"open_interest": float(m.get("open_interest", 0) or 0)},
                ticker=ticker,
                as_of_ts=now,
            ),
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_spread",
                value=spread,
                metadata={},
                ticker=ticker,
                as_of_ts=now,
            ),
        ]
        return records

    # ------------------------------------------------------------------
    # WebSocket orderbook
    # ------------------------------------------------------------------

    async def _ws_connect(self) -> None:
        """Maintain a persistent WebSocket connection for orderbook updates."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed; skipping WebSocket connection")
            return

        backoff = 1
        while True:
            try:
                async with websockets.connect(config.KALSHI_WS_URL) as ws:
                    logger.info("Kalshi WebSocket connected")
                    backoff = 1
                    # Subscribe to orderbook deltas
                    sub_msg = json.dumps({"id": 1, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"]}})
                    await ws.send(sub_msg)
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            ticker = data.get("market_ticker") or data.get("ticker")
                            if ticker:
                                self._orderbook[ticker] = data
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Kalshi WebSocket error: %s; reconnecting in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def start_websocket(self) -> None:
        """Schedule the WebSocket coroutine as a background task."""
        loop = asyncio.get_event_loop()
        self._ws_task = loop.create_task(self._ws_connect(), name="kalshi-ws")
