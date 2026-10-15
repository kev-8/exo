"""Unit tests for the WGI annual-release ingestor — no network calls.

The two things that must not go wrong: an edition already held is never
re-written (so the job can poll freely), and a newer edition always lands
with rows that outrank the old ones in get_latest().
"""

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from exo.ingestion.wgi import (
    DIMENSIONS,
    ISO2_TO_ISO3,
    WGIIngestor,
    _normalise_estimate,
    parse_dataset,
)
from exo.models import FeatureQuery
from exo.store.feature_store import FeatureStore

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def _dta(rows: list[tuple]) -> bytes:
    """Build a Stata file shaped like the WB release: (iso3, year, dim, estimate)."""
    df = pd.DataFrame(
        [
            {
                "econ_code": iso3, "wgi_year": float(year), "dimension": dim,
                "nsource": 8.0, "estimate": est, "estimate_se": 0.2, "score": 50.0,
            }
            for iso3, year, dim, est in rows
        ]
    )
    buf = io.BytesIO()
    df.to_stata(buf, write_index=False)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def ingestor(store):
    from exo.event_bus import EventBus

    return WGIIngestor(store=store, bus=EventBus())


class _Resp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _client(published: dict[int, bytes]):
    """Fake httpx.AsyncClient serving the given editions; others 404."""

    def edition_of(url: str) -> int:
        return int(url.rsplit("-", 1)[1].split(".")[0])

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.head.side_effect = lambda url: _Resp(200 if edition_of(url) in published else 404)
    client.get.side_effect = lambda url: _Resp(200, published[edition_of(url)])
    return client


async def _run(ingestor, published, now=NOW):
    with patch("exo.ingestion.wgi.httpx.AsyncClient", return_value=_client(published)), \
         patch.object(ingestor, "utcnow", return_value=now):
        return await ingestor.run()


class TestParse:
    def test_latest_year_per_dimension(self):
        blob = _dta([("USA", 2024, "pv", 0.1), ("USA", 2025, "pv", -0.3), ("USA", 2024, "va", 1.0)])
        out = parse_dataset(blob, {"USA": "US"})
        assert out["US"]["pv"]["wgi_year"] == 2025
        assert out["US"]["pv"]["estimate"] == pytest.approx(-0.3)
        # va's newest year is 2024; it isn't dropped for lacking 2025.
        assert out["US"]["va"]["wgi_year"] == 2024

    def test_missing_estimate_falls_back_to_previous_year(self):
        blob = _dta([("HTI", 2024, "pv", -1.4), ("HTI", 2025, "pv", float("nan"))])
        assert parse_dataset(blob, {"HTI": "HT"})["HT"]["pv"]["wgi_year"] == 2024

    def test_untracked_economies_and_dimensions_ignored(self):
        blob = _dta([("DEU", 2025, "pv", 0.5), ("USA", 2025, "ge", 1.2)])
        assert parse_dataset(blob, {"USA": "US"}) == {}

    def test_country_map_covers_divergent_codes(self):
        assert ISO2_TO_ISO3["KP"] == "PRK"
        assert ISO2_TO_ISO3["TW"] == "TWN"
        assert len(ISO2_TO_ISO3) == 20


class TestNormalise:
    def test_scale(self):
        assert _normalise_estimate(0.0) == pytest.approx(0.5)
        assert _normalise_estimate(2.5) == pytest.approx(1.0)
        assert _normalise_estimate(-2.5) == pytest.approx(0.0)
        # The new methodology can exceed the nominal range; clamp.
        assert _normalise_estimate(-3.1) == 0.0

    def test_signal_names_match_dimension_scorer(self):
        assert set(DIMENSIONS.values()) == {
            "political_stability", "voice_accountability",
            "rule_of_law", "control_of_corruption",
        }


class TestRun:
    @pytest.mark.asyncio
    async def test_writes_newest_edition_stamped_at_ingest(self, ingestor, store):
        written = await _run(ingestor, {
            2026: _dta([("USA", 2025, d, 0.5) for d in DIMENSIONS]),
        })
        assert len(written) == 4
        rec = store.get_latest(entity="US", signal_type="political_stability", source="wgi")
        assert rec.value == pytest.approx(0.6)
        assert rec.metadata["edition"] == 2026
        assert rec.metadata["wgi_year"] == 2025
        assert rec.as_of_ts == NOW

    @pytest.mark.asyncio
    async def test_falls_back_to_last_years_edition(self, ingestor):
        written = await _run(ingestor, {2025: _dta([("USA", 2024, "pv", 0.0)])})
        assert written[0].metadata["edition"] == 2025

    @pytest.mark.asyncio
    async def test_held_edition_is_not_rewritten(self, ingestor):
        published = {2026: _dta([("USA", 2025, "pv", 0.5)])}
        assert await _run(ingestor, published)
        client = _client(published)
        with patch("exo.ingestion.wgi.httpx.AsyncClient", return_value=client), \
             patch.object(ingestor, "utcnow", return_value=NOW + timedelta(days=7)):
            assert await ingestor.run() == []
        client.get.assert_not_called()  # one HEAD, no download

    @pytest.mark.asyncio
    async def test_newer_edition_outranks_previous(self, ingestor, store):
        await _run(ingestor, {2026: _dta([("USA", 2025, "pv", 0.5)])})
        later = NOW + timedelta(days=365)
        await _run(ingestor, {
            2026: _dta([("USA", 2025, "pv", 0.5)]),
            2027: _dta([("USA", 2026, "pv", -0.5)]),
        }, now=later)
        rec = store.get_latest(entity="US", signal_type="political_stability", source="wgi")
        assert rec.metadata["edition"] == 2027
        assert rec.value == pytest.approx(0.4)
        # A point-in-time read before the second ingest still sees edition 2026.
        old = store.read(FeatureQuery(entity="US", signal_type="political_stability",
                                      source="wgi", as_of_ts=later - timedelta(days=1)))
        assert {r.metadata["edition"] for r in old} == {2026}

    @pytest.mark.asyncio
    async def test_no_edition_published_writes_nothing(self, ingestor):
        assert await _run(ingestor, {}) == []

    @pytest.mark.asyncio
    async def test_unreadable_store_does_not_trigger_download(self, ingestor):
        client = _client({2026: _dta([("USA", 2025, "pv", 0.5)])})
        with patch("exo.ingestion.wgi.httpx.AsyncClient", return_value=client), \
             patch.object(ingestor.store, "latest_metadata_by_entity",
                          side_effect=RuntimeError("boom")):
            assert await ingestor.run() == []
        client.get.assert_not_called()
