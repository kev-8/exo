"""Integration tests for FeatureStore — round-trip and anti-lookahead."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from exo.models import FeatureQuery, FeatureRecord
from exo.store.feature_store import FeatureStore


@pytest.fixture
def tmp_store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def backtest_store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None, backtest_mode=True)


def make_record(source="gdelt", entity="US", signal_type="news_sentiment", value=0.5,
                as_of_ts=None) -> FeatureRecord:
    return FeatureRecord(
        source=source,
        entity=entity,
        signal_type=signal_type,
        value=value,
        metadata={"test": True},
        as_of_ts=as_of_ts or datetime.now(timezone.utc),
    )


class TestFeatureStoreRoundTrip:
    def test_write_and_read_back(self, tmp_store):
        now = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        rec = make_record(as_of_ts=now)
        tmp_store.write(rec)

        results = tmp_store.read(FeatureQuery(entity="US", signal_type="news_sentiment"))
        assert len(results) == 1
        assert abs(results[0].value - 0.5) < 1e-6
        assert results[0].source == "gdelt"

    def test_get_latest(self, tmp_store):
        t1 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)
        tmp_store.write(make_record(value=0.3, as_of_ts=t1))
        tmp_store.write(make_record(value=0.7, as_of_ts=t2))

        latest = tmp_store.get_latest("US", "news_sentiment", as_of_ts=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc))
        assert latest is not None
        assert abs(latest.value - 0.7) < 1e-6

    def test_write_multiple_sources(self, tmp_store):
        now = datetime.now(timezone.utc)
        tmp_store.write(make_record(source="gdelt", as_of_ts=now))
        tmp_store.write(make_record(source="fred", signal_type="economic_indicator", as_of_ts=now))

        gdelt = tmp_store.read(FeatureQuery(source="gdelt"))
        fred = tmp_store.read(FeatureQuery(source="fred"))
        assert len(gdelt) == 1
        assert len(fred) == 1

    def test_ticker_query(self, tmp_store):
        now = datetime.now(timezone.utc)
        rec = FeatureRecord(
            source="kalshi", entity="KXTEST-1", signal_type="market_price",
            value=0.55, metadata={}, as_of_ts=now, ticker="KXTEST-1",
        )
        tmp_store.write(rec)
        results = tmp_store.get_for_ticker("KXTEST-1")
        assert len(results) == 1
        assert results[0].ticker == "KXTEST-1"

    def test_batch_write(self, tmp_store):
        now = datetime.now(timezone.utc)
        records = [make_record(value=float(i) / 10, as_of_ts=now) for i in range(5)]
        tmp_store.write_batch(records)
        results = tmp_store.read(FeatureQuery(entity="US", limit=10))
        assert len(results) == 5


class TestAntiLookahead:
    def test_as_of_ts_filters_future(self, tmp_store):
        t_past = datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)
        t_future = datetime(2025, 1, 20, 12, 0, tzinfo=timezone.utc)
        t_query = datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc)

        tmp_store.write(make_record(value=0.3, as_of_ts=t_past))
        tmp_store.write(make_record(value=0.8, as_of_ts=t_future))

        results = tmp_store.read(FeatureQuery(entity="US", as_of_ts=t_query))
        assert len(results) == 1
        assert abs(results[0].value - 0.3) < 1e-6

    def test_backtest_mode_raises_without_as_of_ts(self, backtest_store):
        backtest_store.write(make_record())
        with pytest.raises(ValueError, match="as_of_ts"):
            backtest_store.read(FeatureQuery(entity="US"))

    def test_backtest_mode_get_latest_raises(self, backtest_store):
        backtest_store.write(make_record())
        with pytest.raises(ValueError, match="as_of_ts"):
            backtest_store.get_latest("US", "news_sentiment")

    def test_backtest_mode_passes_with_as_of_ts(self, backtest_store):
        now = datetime.now(timezone.utc)
        backtest_store.write(make_record(as_of_ts=now - timedelta(hours=1)))
        result = backtest_store.get_latest("US", "news_sentiment", as_of_ts=now)
        assert result is not None


class TestDatetimeParsing:
    def test_microsecond_timestamps_roundtrip(self, tmp_store):
        """Records with microseconds in as_of_ts must survive a write/read cycle."""
        import re
        ts = datetime(2026, 4, 6, 2, 0, 58, 937052, tzinfo=timezone.utc)
        assert ts.microsecond == 937052
        rec = make_record(as_of_ts=ts)
        tmp_store.write(rec)
        results = tmp_store.read(FeatureQuery(entity="US", signal_type="news_sentiment"))
        assert len(results) == 1
        assert results[0].as_of_ts.microsecond == 937052

    def test_mixed_microsecond_and_whole_second_timestamps(self, tmp_store):
        """Store must handle records with and without microseconds in the same partition."""
        t1 = datetime(2026, 1, 1, 10, 0, 0, 0, tzinfo=timezone.utc)        # no microseconds
        t2 = datetime(2026, 1, 1, 11, 0, 0, 500000, tzinfo=timezone.utc)   # with microseconds
        tmp_store.write(make_record(value=0.1, as_of_ts=t1))
        tmp_store.write(make_record(value=0.9, as_of_ts=t2))
        results = tmp_store.read(FeatureQuery(entity="US", signal_type="news_sentiment"))
        assert len(results) == 2
