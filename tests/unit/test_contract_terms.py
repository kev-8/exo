"""Tests for contract settlement-terms capture on the Kalshi ingestor.

Terms are a property of the contract, not an observation of it, so the
behaviour that matters is: emitted once, re-emitted when Kalshi amends them,
never duplicated across the 96 daily price snapshots, and not lost if a write
fails.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from exo.event_bus import EventBus
from exo.models import FeatureQuery
from exo.store.feature_store import FeatureStore

TERMS_SIGNAL = "contract_terms"


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def ingestor(store):
    from exo.ingestion.kalshi import KalshiIngestor
    return KalshiIngestor(store=store, bus=EventBus())


def _market(ticker="KXEURUSD-26OCT06", rules="If the open price is above 1.134, YES.", **kw):
    m = {
        "ticker": ticker,
        "title": "EUR/USD above 1.134",
        "event_ticker": "KXEURUSD-26OCT06",
        "status": "active",
        "close_time": "2026-10-06T14:00:00Z",
        "yes_bid_dollars": 0.42,
        "yes_ask_dollars": 0.45,
        "no_bid_dollars": 0.55,
        "no_ask_dollars": 0.58,
        "volume_24h_fp": "500",
        "volume_fp": "9000",
        "open_interest_fp": "120",
        "yes_bid_size_fp": "30",
        "yes_ask_size_fp": "40",
        "rules_primary": rules,
        "can_close_early": True,
        "strike_type": "greater",
        "floor_strike": 1.134,
        "yes_sub_title": "1.13400 or above",
        "market_type": "binary",
        "settlement_timer_seconds": 60,
    }
    m.update(kw)
    return m


def _counts(records):
    from collections import Counter
    return Counter(r.signal_type for r in records)


class TestTermsFingerprint:
    def test_same_terms_same_hash(self, ingestor):
        a = ingestor._terms(_market())
        b = ingestor._terms(_market())
        assert ingestor._terms_fingerprint(a) == ingestor._terms_fingerprint(b)

    def test_amended_rules_change_the_hash(self, ingestor):
        a = ingestor._terms(_market(rules="Original wording."))
        b = ingestor._terms(_market(rules="Amended wording."))
        assert ingestor._terms_fingerprint(a) != ingestor._terms_fingerprint(b)

    def test_price_movement_does_not_change_the_hash(self, ingestor):
        """Terms must not be re-emitted just because the quote moved."""
        a = ingestor._terms(_market(yes_bid_dollars=0.42))
        b = ingestor._terms(_market(yes_bid_dollars=0.91))
        assert ingestor._terms_fingerprint(a) == ingestor._terms_fingerprint(b)

    def test_empty_and_absent_fields_are_omitted(self, ingestor):
        terms = ingestor._terms(_market(rules_secondary="", cap_strike=None))
        assert "rules_secondary" not in terms
        assert "cap_strike" not in terms
        assert terms["rules_primary"]


class TestEmission:
    @pytest.mark.asyncio
    async def test_terms_emitted_once_then_suppressed(self, ingestor):
        """96 polls a day must not write 96 identical terms rows."""
        ingestor._series_category = {"KXEURUSD": "Financials"}
        ingestor._series_refreshed_at = _now()
        ingestor._active_series = {"KXEURUSD"}
        ingestor._active_refreshed_at = _now()

        with patch.object(ingestor, "_probe_series",
                          new=AsyncMock(return_value=[_market()])):
            first = await ingestor.run()
            second = await ingestor.run()
            third = await ingestor.run()

        assert _counts(first)[TERMS_SIGNAL] == 1
        assert _counts(second)[TERMS_SIGNAL] == 0
        assert _counts(third)[TERMS_SIGNAL] == 0

        # Price rows keep flowing every cycle — only terms are deduped.
        assert _counts(second)["market_price"] == 1
        assert _counts(third)["market_price"] == 1

        rows = ingestor.store.read(FeatureQuery(
            source="kalshi", signal_type=TERMS_SIGNAL))
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_amended_rules_emit_a_new_row(self, ingestor):
        ingestor._series_category = {"KXEURUSD": "Financials"}
        ingestor._series_refreshed_at = _now()
        ingestor._active_series = {"KXEURUSD"}
        ingestor._active_refreshed_at = _now()

        with patch.object(ingestor, "_probe_series",
                          new=AsyncMock(return_value=[_market(rules="Original.")])):
            await ingestor.run()
        with patch.object(ingestor, "_probe_series",
                          new=AsyncMock(return_value=[_market(rules="Amended.")])):
            again = await ingestor.run()

        assert _counts(again)[TERMS_SIGNAL] == 1
        rows = ingestor.store.read(FeatureQuery(
            source="kalshi", signal_type=TERMS_SIGNAL))
        assert len(rows) == 2
        assert {r.metadata["rules_primary"] for r in rows} == {"Original.", "Amended."}

    @pytest.mark.asyncio
    async def test_terms_cache_survives_a_restart(self, store):
        from exo.ingestion.kalshi import KalshiIngestor

        def _fresh():
            ing = KalshiIngestor(store=store, bus=EventBus())
            ing._series_category = {"KXEURUSD": "Financials"}
            ing._series_refreshed_at = _now()
            ing._active_series = {"KXEURUSD"}
            ing._active_refreshed_at = _now()
            return ing

        with patch.object(KalshiIngestor, "_probe_series",
                          new=AsyncMock(return_value=[_market()])):
            assert _counts(await _fresh().run())[TERMS_SIGNAL] == 1
            assert _counts(await _fresh().run())[TERMS_SIGNAL] == 0

    @pytest.mark.asyncio
    async def test_failed_write_lets_terms_be_re_emitted(self, ingestor):
        ingestor._series_category = {"KXEURUSD": "Financials"}
        ingestor._series_refreshed_at = _now()
        ingestor._active_series = {"KXEURUSD"}
        ingestor._active_refreshed_at = _now()

        with patch.object(ingestor, "_probe_series",
                          new=AsyncMock(return_value=[_market()])):
            with patch.object(ingestor.store, "write_batch",
                              side_effect=OSError("disk full")):
                assert await ingestor.run() == []
            recovered = await ingestor.run()

        assert _counts(recovered)[TERMS_SIGNAL] == 1

    def test_metadata_carries_the_rules_and_strike(self, ingestor):
        from exo.models import RawRecord

        raw = RawRecord(source="kalshi", entity="KXEURUSD-26OCT06",
                        raw={**_market(), "category": "Financials",
                             "series_ticker": "KXEURUSD",
                             "_emit_terms": True, "_terms_hash": "abc123"},
                        fetched_at=_now())
        terms = [r for r in ingestor.normalise(raw) if r.signal_type == TERMS_SIGNAL]
        assert len(terms) == 1
        md = terms[0].metadata
        assert md["rules_primary"].startswith("If the open price")
        assert md["strike_type"] == "greater"
        assert md["floor_strike"] == 1.134
        assert md["can_close_early"] is True
        assert md["terms_hash"] == "abc123"
        assert md["category"] == "Financials"
        assert terms[0].value == float(len(ingestor._terms(_market())))

    def test_no_terms_fields_means_no_terms_row(self, ingestor):
        from exo.models import RawRecord

        bare = {k: v for k, v in _market().items()
                if k not in ("rules_primary", "can_close_early", "strike_type",
                             "floor_strike", "yes_sub_title", "market_type",
                             "settlement_timer_seconds")}
        raw = RawRecord(source="kalshi", entity="X",
                        raw={**bare, "_emit_terms": False}, fetched_at=_now())
        assert not [r for r in ingestor.normalise(raw) if r.signal_type == TERMS_SIGNAL]


class TestLatestMetadataByEntity:
    def test_returns_newest_value_per_entity(self, store):
        from datetime import timedelta
        from exo.models import FeatureRecord

        t0 = _now() - timedelta(days=2)
        store.write_batch([
            FeatureRecord(source="s", entity="A", signal_type="t", value=1.0,
                          metadata={"terms_hash": "old"}, as_of_ts=t0),
            FeatureRecord(source="s", entity="A", signal_type="t", value=1.0,
                          metadata={"terms_hash": "new"}, as_of_ts=_now()),
            FeatureRecord(source="s", entity="B", signal_type="t", value=1.0,
                          metadata={"terms_hash": "bee"}, as_of_ts=_now()),
            FeatureRecord(source="s", entity="C", signal_type="other", value=1.0,
                          metadata={"terms_hash": "nope"}, as_of_ts=_now()),
        ])
        assert store.latest_metadata_by_entity("s", "t", "terms_hash") == {
            "A": "new", "B": "bee"}

    def test_missing_source_is_empty(self, store):
        assert store.latest_metadata_by_entity("none", "t", "k") == {}
