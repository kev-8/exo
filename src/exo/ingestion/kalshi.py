"""Kalshi ingestor — RSA-PSS auth, REST metadata.

REST polls market metadata every 15 minutes, scoped to a category-filtered
set of series (see _refresh_focused_series / _refresh_active_series below).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, KalshiMarket, RawRecord

logger = logging.getLogger(__name__)

# How often to re-pull the full series catalog (rarely changes — new series
# are added occasionally, not continuously).
_SERIES_REFRESH_INTERVAL = timedelta(hours=24)

# How often to re-probe *every* focused series for open markets, to catch
# series that just became active. Between refreshes, only the already-active
# subset is polled (cheap — most focused series have nothing open at any
# given moment).
_ACTIVE_REFRESH_INTERVAL = timedelta(hours=1)

# Max concurrent /markets requests during a series probe. Kept modest to
# stay well under Kalshi's rate limit even when probing all ~6k series.
_PROBE_CONCURRENCY = 20

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
    """Ingest Kalshi market data via REST."""

    source = "kalshi"

    def __init__(
        self,
        tickers: list[str] | None = None,
        categories: set[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tickers = tickers or []
        self.categories = categories or config.FOCUSED_CATEGORIES

        # series_ticker -> category, for every series in self.categories.
        self._series_category: dict[str, str] = {}
        self._series_refreshed_at: datetime | None = None

        # Subset of _series_category currently believed to have open markets.
        self._active_series: set[str] = set()
        self._active_refreshed_at: datetime | None = None

    # ------------------------------------------------------------------
    # Series discovery
    # ------------------------------------------------------------------

    async def _refresh_focused_series(self, client: httpx.AsyncClient) -> None:
        """Rebuild the series -> category allowlist from /series.

        A single unpaginated call returns the entire series catalog (~14k
        entries); filtering client-side to self.categories is far cheaper
        than discovering relevant series by paging through /markets or
        /events, both of which are dominated by high-frequency synthetic
        markets unrelated to these categories.
        """
        path = "/series"
        headers = _build_auth_headers("GET", "/trade-api/v2" + path)
        try:
            resp = await client.get(config.KALSHI_BASE_URL + path, headers=headers)
            resp.raise_for_status()
            all_series = resp.json().get("series", [])
        except Exception as exc:
            logger.warning("Kalshi /series refresh failed: %s", exc)
            return

        self._series_category = {
            s["ticker"]: s["category"]
            for s in all_series
            if s.get("ticker") and s.get("category") in self.categories
        }
        self._series_refreshed_at = self.utcnow()
        logger.info(
            "Kalshi focused-series allowlist refreshed: %d series across %d categories",
            len(self._series_category), len(self.categories),
        )

    async def _probe_series(
        self, client: httpx.AsyncClient, series_ticker: str, sem: asyncio.Semaphore
    ) -> list[dict]:
        """Return currently open markets for one series (empty if none/error)."""
        async with sem:
            path = "/markets"
            headers = _build_auth_headers("GET", "/trade-api/v2" + path)
            try:
                resp = await client.get(
                    config.KALSHI_BASE_URL + path,
                    headers=headers,
                    params={"series_ticker": series_ticker, "status": "open", "limit": 200},
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.debug("Kalshi rate limited on series=%s; sleeping %ds", series_ticker, retry_after)
                    await asyncio.sleep(retry_after)
                    return []
                resp.raise_for_status()
                return resp.json().get("markets", [])
            except Exception as exc:
                logger.debug("Kalshi series probe failed for %s: %s", series_ticker, exc)
                return []

    async def _refresh_active_series(self, client: httpx.AsyncClient) -> dict[str, list[dict]]:
        """Probe every focused series for open markets.

        This call itself carries live pricing, so its result is used
        directly as this cycle's market data rather than being fetched
        again — it's the expensive (~len(series_category) requests) sweep,
        run at most every _ACTIVE_REFRESH_INTERVAL.
        """
        sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
        tickers = list(self._series_category.keys())
        results = await asyncio.gather(*(self._probe_series(client, t, sem) for t in tickers))
        by_series = {t: markets for t, markets in zip(tickers, results) if markets}
        self._active_series = set(by_series.keys())
        self._active_refreshed_at = self.utcnow()
        logger.info(
            "Kalshi active-series refresh: %d/%d focused series have open markets",
            len(self._active_series), len(tickers),
        )
        return by_series

    # ------------------------------------------------------------------
    # REST fetch
    # ------------------------------------------------------------------

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            if self._series_refreshed_at is None or now - self._series_refreshed_at > _SERIES_REFRESH_INTERVAL:
                await self._refresh_focused_series(client)

            if not self._series_category:
                logger.warning("Kalshi focused-series allowlist is empty; skipping poll cycle")
                return raws

            if self._active_refreshed_at is None or now - self._active_refreshed_at > _ACTIVE_REFRESH_INTERVAL:
                by_series = await self._refresh_active_series(client)
            else:
                sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
                tickers = list(self._active_series)
                results = await asyncio.gather(*(self._probe_series(client, t, sem) for t in tickers))
                by_series = dict(zip(tickers, results))
                # Markets close between polls — drop series with nothing open
                # this cycle so the next fast poll doesn't keep probing them.
                self._active_series = {t for t, markets in by_series.items() if markets}

            for series_ticker, markets in by_series.items():
                category = self._series_category.get(series_ticker, "")
                for m in markets:
                    raw = {**m, "category": category}
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity=m.get("ticker", ""),
                            raw=raw,
                            fetched_at=now,
                        )
                    )

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
                    "category": m.get("category", ""),
                    "event_ticker": m.get("event_ticker", ""),
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

