"""Unit tests for the Kalshi settlement archive ingestor.

The archive's whole value is that a label written once is never lost and never
duplicated, so the tests concentrate on the dedupe invariant and on dating the
row by settlement rather than by the sweep that found it.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from exo.models import FeatureQuery, RawRecord
from exo.store.feature_store import FeatureStore


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def ingestor(store):
    from exo.ingestion.kalshi_resolution import KalshiResolutionIngestor
    from exo.event_bus import EventBus

    return KalshiResolutionIngestor(store=store, bus=EventBus())


def _market(ticker="KXUSDJPY-26SEP30", result="yes", close="2026-09-30T20:00:00Z", **kw):
    return {
        "ticker": ticker,
        "result": result,
        "close_time": close,
        "event_ticker": "KXUSDJPY-26SEP30",
        "title": "USD/JPY above 150",
        "volume_fp": "1234",
        "open_interest_fp": "567",
        **kw,
    }


class TestNormalise:
    def test_yes_is_one_no_is_zero(self, ingestor):
        yes = ingestor.normalise(RawRecord(
            source="kalshi_resolution", entity="T1",
            raw=_market(ticker="T1", result="yes"), fetched_at=_now()))
        no = ingestor.normalise(RawRecord(
            source="kalshi_resolution", entity="T2",
            raw=_market(ticker="T2", result="no"), fetched_at=_now()))
        assert yes[0].value == 1.0
        assert no[0].value == 0.0
        assert yes[0].signal_type == "market_resolution"

    def test_row_is_dated_by_settlement_not_by_sweep(self, ingestor):
        """as_of_ts must be close_time, else a point-in-time read sees the
        label before the outcome existed."""
        swept_at = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
        rec = ingestor.normalise(RawRecord(
            source="kalshi_resolution", entity="T1",
            raw=_market(close="2026-09-30T20:00:00Z"), fetched_at=swept_at))[0]
        assert rec.as_of_ts == datetime(2026, 9, 30, 20, 0, tzinfo=timezone.utc)

    def test_metadata_carries_category_and_liquidity(self, ingestor):
        rec = ingestor.normalise(RawRecord(
            source="kalshi_resolution", entity="T1",
            raw=_market(category="Financials", series_ticker="KXUSDJPY"),
            fetched_at=_now()))[0]
        assert rec.metadata["category"] == "Financials"
        assert rec.metadata["series_ticker"] == "KXUSDJPY"
        assert rec.metadata["volume_lifetime"] == 1234.0
        assert rec.metadata["open_interest"] == 567.0

    def test_unparseable_close_time_falls_back_to_fetch_time(self, ingestor):
        swept_at = _now()
        rec = ingestor.normalise(RawRecord(
            source="kalshi_resolution", entity="T1",
            raw=_market(close=None), fetched_at=swept_at))[0]
        assert rec.as_of_ts == swept_at


class TestDedupe:
    @pytest.mark.asyncio
    async def test_second_sweep_writes_nothing_new(self, ingestor):
        """The same settled market seen twice must be stored once."""
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()

        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            first = await ingestor.run()
            second = await ingestor.run()

        assert len(first) == 1
        assert second == []

        rows = ingestor.store.read(FeatureQuery(
            source="kalshi_resolution", signal_type="market_resolution"))
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_seen_set_survives_a_restart(self, store):
        """A fresh ingestor instance must not re-append the existing archive."""
        from exo.event_bus import EventBus
        from exo.ingestion.kalshi_resolution import KalshiResolutionIngestor

        first = KalshiResolutionIngestor(store=store, bus=EventBus())
        first._series_category = {"KXUSDJPY": "Financials"}
        first._series_refreshed_at = _now()
        with patch.object(first, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            assert len(await first.run()) == 1

        restarted = KalshiResolutionIngestor(store=store, bus=EventBus())
        restarted._series_category = {"KXUSDJPY": "Financials"}
        restarted._series_refreshed_at = _now()
        with patch.object(restarted, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            assert await restarted.run() == []

    @pytest.mark.asyncio
    async def test_settled_but_unresolved_is_deferred_not_recorded(self, ingestor):
        """A market settled without a posted result must stay eligible."""
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()

        pending = _market(result="")
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[pending])):
            assert await ingestor.run() == []

        # Result posted on a later sweep — now it should land.
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market(result="yes")])):
            assert len(await ingestor.run()) == 1

    def test_unreadable_store_raises_rather_than_claiming_empty(self, ingestor):
        """An empty seen-set means 'archive everything'; a read failure must
        not be allowed to look like one."""
        with patch.object(ingestor.store, "distinct_entities",
                          side_effect=RuntimeError("parquet unreadable")):
            with pytest.raises(RuntimeError):
                ingestor._ensure_seen()


class TestDistinctEntities:
    def test_returns_entities_for_the_signal_type(self, store):
        from exo.models import FeatureRecord

        store.write_batch([
            FeatureRecord(source="s", entity="A", signal_type="x", value=1.0,
                          metadata={}, as_of_ts=_now()),
            FeatureRecord(source="s", entity="A", signal_type="x", value=1.0,
                          metadata={}, as_of_ts=_now() - timedelta(days=1)),
            FeatureRecord(source="s", entity="B", signal_type="x", value=1.0,
                          metadata={}, as_of_ts=_now()),
            FeatureRecord(source="s", entity="C", signal_type="y", value=1.0,
                          metadata={}, as_of_ts=_now()),
        ])
        assert store.distinct_entities("s", "x") == {"A", "B"}
        assert store.distinct_entities("s") == {"A", "B", "C"}

    def test_missing_source_is_empty_not_an_error(self, store):
        assert store.distinct_entities("never_written") == set()


class TestPagination:
    """The 200-row page cap must not silently truncate the first backfill."""

    @staticmethod
    def _client_returning(pages):
        """An httpx-ish client whose GET walks *pages* in order."""
        from unittest.mock import MagicMock

        calls = {"n": 0}

        async def get(url, headers=None, params=None):
            i = calls["n"]
            calls["n"] += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            resp.json = lambda: pages[i] if i < len(pages) else {"markets": []}
            return resp

        client = MagicMock()
        client.get = get
        return client, calls

    @pytest.mark.asyncio
    async def test_pages_until_cursor_runs_out(self, ingestor):
        import asyncio as _asyncio

        pages = [
            {"markets": [_market(ticker=f"A{i}") for i in range(3)], "cursor": "c1"},
            {"markets": [_market(ticker=f"B{i}") for i in range(3)], "cursor": None},
        ]
        client, calls = self._client_returning(pages)
        got = await ingestor._probe_settled(client, "KXUSDJPY", _asyncio.Semaphore(1), set())

        assert len(got) == 6
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_stops_early_when_a_page_is_all_already_seen(self, ingestor):
        """Steady-state sweeps must stay at one request per series."""
        import asyncio as _asyncio

        seen = {"A0", "A1", "A2"}
        pages = [
            {"markets": [_market(ticker=t) for t in sorted(seen)], "cursor": "c1"},
            {"markets": [_market(ticker="OLD")], "cursor": None},
        ]
        client, calls = self._client_returning(pages)
        got = await ingestor._probe_settled(client, "KXUSDJPY", _asyncio.Semaphore(1), seen)

        assert calls["n"] == 1, "paged past an all-seen page"
        assert len(got) == 3

    @pytest.mark.asyncio
    async def test_keeps_paging_while_pages_contain_new_tickers(self, ingestor):
        import asyncio as _asyncio

        seen = {"A0"}
        pages = [
            {"markets": [_market(ticker="A0"), _market(ticker="NEW1")], "cursor": "c1"},
            {"markets": [_market(ticker="NEW2")], "cursor": None},
        ]
        client, calls = self._client_returning(pages)
        got = await ingestor._probe_settled(client, "KXUSDJPY", _asyncio.Semaphore(1), seen)

        assert calls["n"] == 2
        assert len(got) == 3


class TestWriteFailureDoesNotLoseLabels:
    """A settlement is a one-time label: if the write fails it must be retried.

    BaseIngestor.run logs a write_batch failure and returns an empty list, so
    without advancing the seen-set only on success, a dropped batch would be
    skipped on every subsequent sweep and the label lost for good.
    """

    @pytest.mark.asyncio
    async def test_failed_write_is_retried_on_the_next_sweep(self, ingestor):
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()

        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            with patch.object(ingestor.store, "write_batch",
                              side_effect=OSError("disk full")):
                assert await ingestor.run() == []

            # The label was never persisted, so it must still be pending.
            assert "KXUSDJPY-26SEP30" not in ingestor._ensure_seen()

            recovered = await ingestor.run()

        assert [r.entity for r in recovered] == ["KXUSDJPY-26SEP30"]
        rows = ingestor.store.read(FeatureQuery(
            source="kalshi_resolution", signal_type="market_resolution"))
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_successful_write_advances_the_seen_set(self, ingestor):
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            await ingestor.run()
        assert "KXUSDJPY-26SEP30" in ingestor._ensure_seen()

    @pytest.mark.asyncio
    async def test_rate_limit_retry_does_not_consume_the_page_budget(self, ingestor):
        """A 429 re-reads the same page, so it must not count as a page read."""
        import asyncio as _asyncio
        from unittest.mock import MagicMock

        import exo.ingestion.kalshi_resolution as mod

        # One 429, then two real pages. With the retry counted as a page this
        # would stop after the first real page.
        script = [
            {"status": 429},
            {"markets": [_market(ticker="NEW1")], "cursor": "c1"},
            {"markets": [_market(ticker="NEW2")], "cursor": None},
        ]
        calls = {"n": 0}

        async def get(url, headers=None, params=None):
            step = script[calls["n"]]
            calls["n"] += 1
            resp = MagicMock()
            if step.get("status") == 429:
                resp.status_code = 429
                resp.headers = {"Retry-After": "0"}
                return resp
            resp.status_code = 200
            resp.headers = {}
            resp.raise_for_status = lambda: None
            resp.json = lambda: step
            return resp

        client = MagicMock()
        client.get = get

        with patch.object(mod, "_MAX_PAGES_PER_SERIES", 2):
            got = await ingestor._probe_settled(
                client, "KXUSDJPY", _asyncio.Semaphore(1), set())

        assert calls["n"] == 3
        assert {m["ticker"] for m in got} == {"NEW1", "NEW2"}

    @pytest.mark.asyncio
    async def test_persistent_rate_limiting_gives_up_without_hanging(self, ingestor):
        import asyncio as _asyncio
        from unittest.mock import MagicMock

        calls = {"n": 0}

        async def get(url, headers=None, params=None):
            calls["n"] += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.headers = {"Retry-After": "0"}
            return resp

        client = MagicMock()
        client.get = get

        got = await ingestor._probe_settled(
            client, "KXUSDJPY", _asyncio.Semaphore(1), set())

        assert got == []
        assert calls["n"] <= 5, "retries are not bounded"


class TestNonBinarySettlements:
    """Some markets finalize at a fractional value instead of yes/no.

    A tie splits 50/50 and a proportionally-settled strike ladder lands
    anywhere in (0, 1). Coercing those to a binary label would invent an
    outcome, so no row is written — but they are finalized, not pending, and
    the sweep must not report them as if a later sweep will resolve them.
    """

    @staticmethod
    def _scalar(ticker="KXHORMUZMAX-1", settled="0.5000", notional="1.0000"):
        m = _market(ticker=ticker, result="scalar")
        m["status"] = "finalized"
        m["settlement_value_dollars"] = settled
        m["notional_value_dollars"] = notional
        return m

    @pytest.mark.asyncio
    async def test_tie_is_archived_at_its_fraction_not_rounded(self, ingestor):
        """A $0.50 tie must be stored as 0.5, never coerced to 0 or 1."""
        ingestor._series_category = {"KXHORMUZMAX": "World"}
        ingestor._series_refreshed_at = _now()

        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[self._scalar()])):
            written = await ingestor.run()
            assert await ingestor.run() == [], "re-archived on a later sweep"

        assert len(written) == 1
        rec = written[0]
        assert rec.value == 0.5
        assert rec.metadata["is_binary"] is False
        assert rec.metadata["settlement_fraction"] == 0.5
        assert rec.metadata["result"] == "scalar"

    @pytest.mark.asyncio
    async def test_proportional_ladder_keeps_its_magnitude(self, ingestor):
        ingestor._series_category = {"KXULTA": "Financials"}
        ingestor._series_refreshed_at = _now()
        with patch.object(ingestor, "_probe_settled", new=AsyncMock(
                return_value=[self._scalar(ticker="KXULTA-1", settled="0.4400")])):
            written = await ingestor.run()
        assert written[0].value == pytest.approx(0.44)

    @pytest.mark.asyncio
    async def test_binary_rows_are_still_flagged_binary(self, ingestor):
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market(result="yes")])):
            written = await ingestor.run()
        assert written[0].value == 1.0
        assert written[0].metadata["is_binary"] is True

    @pytest.mark.asyncio
    async def test_finalized_without_a_settlement_value_is_deferred(self, ingestor):
        """Nothing can be said about the outcome, so don't invent a 0.0."""
        ingestor._series_category = {"S": "World"}
        ingestor._series_refreshed_at = _now()
        m = _market(ticker="X-1", result="scalar")
        m["status"] = "finalized"
        m.pop("settlement_value_dollars", None)
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[m])):
            assert await ingestor.run() == []

    @pytest.mark.asyncio
    async def test_zero_notional_does_not_divide_by_zero(self, ingestor):
        ingestor._series_category = {"S": "World"}
        ingestor._series_refreshed_at = _now()
        with patch.object(ingestor, "_probe_settled", new=AsyncMock(
                return_value=[self._scalar(settled="0.5", notional="0")])):
            assert await ingestor.run() == []

    @pytest.mark.asyncio
    async def test_pending_result_is_still_picked_up_later(self, ingestor):
        """A blank result must not be lumped in with the finalized ones."""
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()

        pending = _market(result="")
        pending["status"] = "settled"
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[pending])):
            assert await ingestor.run() == []
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market(result="yes")])):
            assert len(await ingestor.run()) == 1

    @pytest.mark.asyncio
    async def test_counts_are_reported_separately(self, ingestor, caplog):
        import logging as _logging

        ingestor._series_category = {"S": "World"}
        ingestor._series_refreshed_at = _now()

        scalar = _market(ticker="A-1", result="scalar")
        scalar["status"] = "finalized"
        scalar["settlement_value_dollars"] = "0.5000"
        scalar["notional_value_dollars"] = "1.0000"
        pending = _market(ticker="B-1", result="")
        pending["status"] = "settled"

        with caplog.at_level(_logging.INFO, logger="exo.ingestion.kalshi_resolution"):
            with patch.object(ingestor, "_probe_settled",
                              new=AsyncMock(return_value=[scalar, pending])):
                await ingestor.run()

        msg = "\n".join(r.getMessage() for r in caplog.records)
        assert "1 of them non-binary fractional settlements" in msg
        assert "1 deferred awaiting a posted result" in msg


class TestConcurrentWriters:
    """The in-memory seen-set must not assume this process is the only writer.

    A one-off backfill run while the scheduler was up produced 504 duplicate
    rows: the scheduler had cached its seen-set hours earlier and re-archived
    everything the backfill had just written.
    """

    @pytest.mark.asyncio
    async def test_rows_written_by_another_process_are_not_re_archived(self, store):
        from exo.event_bus import EventBus
        from exo.ingestion.kalshi_resolution import KalshiResolutionIngestor

        scheduler = KalshiResolutionIngestor(store=store, bus=EventBus())
        scheduler._series_category = {"KXUSDJPY": "Financials"}
        scheduler._series_refreshed_at = _now()
        # Seeds the cache against an empty archive, as a long-lived process does.
        assert scheduler._ensure_seen() == set()

        # A separate process archives the contract in the meantime.
        backfill = KalshiResolutionIngestor(store=store, bus=EventBus())
        backfill._series_category = {"KXUSDJPY": "Financials"}
        backfill._series_refreshed_at = _now()
        with patch.object(backfill, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            assert len(await backfill.run()) == 1

        with patch.object(scheduler, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            assert await scheduler.run() == [], "re-archived another writer's row"

        rows = store.read(FeatureQuery(
            source="kalshi_resolution", signal_type="market_resolution"))
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_own_writes_survive_the_refresh(self, ingestor):
        """Unioning, not replacing — this process's writes predate a fresh read."""
        ingestor._series_category = {"KXUSDJPY": "Financials"}
        ingestor._series_refreshed_at = _now()
        with patch.object(ingestor, "_probe_settled",
                          new=AsyncMock(return_value=[_market()])):
            await ingestor.run()
            assert "KXUSDJPY-26SEP30" in ingestor._refresh_seen()
            assert await ingestor.run() == []
