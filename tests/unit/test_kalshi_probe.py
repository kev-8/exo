"""Tests for the Kalshi series probe: auth-key reuse and rate-limit handling.

A rate-limited or failed probe must not be mistaken for "no open markets" —
that silently drops the series from the active set until the next hourly
sweep — and the RSA key must not be re-parsed on every request.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from exo.event_bus import EventBus
from exo.ingestion import kalshi as K
from exo.store.feature_store import FeatureStore


class _Resp:
    def __init__(self, status: int, markets=None):
        self.status_code = status
        self.headers = {"Retry-After": "0"}
        self._markets = markets or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"markets": self._markets}


def _market(ticker):
    return {"ticker": ticker, "yes_bid_dollars": 0.4, "yes_ask_dollars": 0.5}


@pytest.fixture
def ingestor(tmp_path):
    ing = K.KalshiIngestor(
        store=FeatureStore(data_dir=tmp_path / "features", redis_url=None), bus=EventBus()
    )
    ing._series_category = {"A": "Politics", "B": "Politics"}
    ing._series_refreshed_at = datetime.now(timezone.utc)
    ing._active_refreshed_at = datetime.now(timezone.utc)
    ing._active_series = {"A", "B"}
    return ing


def _client(responses: dict[str, list]):
    """Fake client answering each series from a queue of responses."""
    client = AsyncMock()
    client.__aenter__.return_value = client

    async def get(url, headers=None, params=None):
        return responses[params["series_ticker"]].pop(0)

    client.get.side_effect = get
    return client


async def _fetch(ingestor, responses):
    with patch("exo.ingestion.kalshi.httpx.AsyncClient", return_value=_client(responses)), \
         patch("exo.ingestion.kalshi._build_auth_headers", return_value={}):
        return await ingestor.fetch()


class TestProbe:
    @pytest.mark.asyncio
    async def test_rate_limit_is_retried(self, ingestor):
        raws = await _fetch(ingestor, {
            "A": [_Resp(429), _Resp(200, [_market("A-1")])],
            "B": [_Resp(200, [_market("B-1")])],
        })
        assert {r.entity for r in raws} == {"A-1", "B-1"}
        assert ingestor._probe_stats["rate_limited"] == 1
        assert ingestor._probe_stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_failed_probe_keeps_series_active(self, ingestor):
        raws = await _fetch(ingestor, {
            "A": [_Resp(429)] * K._PROBE_MAX_ATTEMPTS,
            "B": [_Resp(200, [])],
        })
        assert raws == []
        # A could not be seen, so it is kept for the next poll; B genuinely
        # has nothing open and is dropped.
        assert ingestor._active_series == {"A"}
        assert ingestor._probe_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_failed_probe_keeps_series_through_full_sweep(self, ingestor):
        ingestor._active_refreshed_at = None
        await _fetch(ingestor, {
            "A": [_Resp(500)],
            "B": [_Resp(200, [_market("B-1")])],
        })
        assert ingestor._active_series == {"A", "B"}


class TestAuthKey:
    def test_key_parsed_once(self):
        pem = "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n"
        K._key_cache = None
        with patch.object(K, "_CRYPTO_AVAILABLE", True), \
             patch.object(K.config, "KALSHI_PRIVATE_KEY", pem), \
             patch.object(K, "serialization", create=True) as ser:
            ser.load_pem_private_key.return_value = object()
            first = K._load_private_key()
            assert K._load_private_key() is first
            assert ser.load_pem_private_key.call_count == 1
        K._key_cache = None
