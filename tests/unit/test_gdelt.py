"""Unit tests for the GDELT export-file ingestor — no network calls.

The failure modes worth guarding are silent ones: a country code read in the
wrong scheme (FIPS vs ISO) produces plausible-looking zeros, and a download
gap mistaken for a coverage drop produces a fake magnitude collapse. Tests
concentrate there.
"""

import io
import zipfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from exo.ingestion.gdelt import (
    COUNTRY_FIPS_MAP,
    GDELTIngestor,
    _HISTORY,
    _SLOT,
    _Tally,
    _parse_export,
    _ratio_to_score,
    _tone_to_risk,
)
from exo.models import RawRecord

LATEST = datetime(2026, 10, 15, 23, 15, tzinfo=timezone.utc)


def _row(country: str, articles: int, tone: float, goldstein: float = 0.0) -> str:
    cols = [""] * 61
    cols[30] = str(goldstein)
    cols[33] = str(articles)
    cols[34] = str(tone)
    cols[53] = country
    return "\t".join(cols)


def _zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20261015231500.export.CSV", "\n".join(rows) + "\n")
    return buf.getvalue()


@pytest.fixture
def ingestor():
    return GDELTIngestor.__new__(GDELTIngestor)


@pytest.fixture
def live_ingestor():
    from exo.event_bus import EventBus

    return GDELTIngestor(store=object(), bus=EventBus())


def _fill(ing: GDELTIngestor, per_slot: dict[str, _Tally], *, skip=lambda s: False):
    """Populate a full week of slots with *per_slot* tallies."""
    slot = LATEST
    while slot > LATEST - _HISTORY:
        if not skip(slot):
            ing._slots[slot] = {iso: _Tally(**vars(t)) for iso, t in per_slot.items()}
        slot -= _SLOT


async def _fetch(ing: GDELTIngestor):
    with patch.object(ing, "_latest_slot", AsyncMock(return_value=LATEST)), \
         patch.object(ing, "_sync_slots", AsyncMock()):
        return await ing.fetch()


class TestParseExport:
    def test_fips_codes_map_to_iso(self):
        fips_to_iso = {"RS": "RU", "CH": "CN"}
        t = _parse_export(_zip([_row("RS", 5, -3.0), _row("CH", 2, 1.0)]), fips_to_iso)
        assert set(t) == {"RU", "CN"}

    def test_iso_code_in_export_is_not_mistaken_for_fips(self):
        # "RU" is Russia in ISO but not a FIPS code GDELT uses for it.
        t = _parse_export(_zip([_row("RU", 5, -3.0)]), {"RS": "RU"})
        assert t == {}

    def test_tone_is_article_weighted(self):
        t = _parse_export(_zip([_row("US", 3, -4.0), _row("US", 1, 4.0)]), {"US": "US"})["US"]
        assert t.articles == 4 and t.events == 2
        assert t.tone_x_articles / t.articles == pytest.approx(-2.0)

    def test_several_fips_codes_fold_into_one_country(self):
        fips_to_iso = {c: "IL" for c in COUNTRY_FIPS_MAP["IL"]}
        t = _parse_export(_zip([_row("IS", 2, 0), _row("GZ", 3, 0), _row("WE", 1, 0)]), fips_to_iso)
        assert t["IL"].articles == 6

    def test_short_and_malformed_rows_skipped(self):
        rows = ["too\tshort", _row("US", 0, 0).replace("\t0\t", "\tx\t", 1), _row("US", 2, -1.0)]
        t = _parse_export(_zip(rows), {"US": "US"})
        assert t["US"].events >= 1

    def test_fips_map_covers_known_divergent_codes(self):
        assert COUNTRY_FIPS_MAP["RU"] == ("RS",)
        assert COUNTRY_FIPS_MAP["NG"] == ("NI",)
        assert COUNTRY_FIPS_MAP["ZA"] == ("SF",)
        assert COUNTRY_FIPS_MAP["GB"] == ("UK",)


class TestScales:
    def test_tone_mapping_matches_doc_api_era(self):
        # Old ingestor: tone_score = tone/10 clamped, value = (1 - tone_score)/2.
        assert _tone_to_risk(-4.0) == pytest.approx(0.70)
        assert _tone_to_risk(0.0) == pytest.approx(0.5)
        assert _tone_to_risk(-50.0) == 1.0
        assert _tone_to_risk(50.0) == 0.0

    def test_normal_volume_is_neutral(self):
        assert _ratio_to_score(1.0) == pytest.approx(0.5)
        assert _ratio_to_score(3.0) == pytest.approx(0.75)
        assert _ratio_to_score(0.0) == 0.0


class TestFetch:
    @pytest.mark.asyncio
    async def test_steady_volume_scores_neutral(self, live_ingestor):
        _fill(live_ingestor, {"US": _Tally(articles=10, events=4, tone_x_articles=-30.0)})
        raws = {r.entity: r for r in await _fetch(live_ingestor)}
        recs = {r.signal_type: r for r in live_ingestor.normalise(raws["US"])}
        assert recs["news_magnitude"].value == pytest.approx(0.5)
        assert recs["news_magnitude"].metadata["articles_24h"] == pytest.approx(960)
        assert recs["news_sentiment"].value == pytest.approx(_tone_to_risk(-3.0))

    @pytest.mark.asyncio
    async def test_spike_raises_magnitude(self, live_ingestor):
        _fill(live_ingestor, {"UA": _Tally(articles=10)})
        for s in list(live_ingestor._slots):
            if s > LATEST - timedelta(hours=24):
                live_ingestor._slots[s] = {"UA": _Tally(articles=40)}
        raw = next(r for r in await _fetch(live_ingestor) if r.entity == "UA")
        mag = next(r for r in live_ingestor.normalise(raw) if r.signal_type == "news_magnitude")
        assert mag.value == pytest.approx(0.8)  # 4x baseline

    @pytest.mark.asyncio
    async def test_missed_slots_do_not_read_as_coverage_drop(self, live_ingestor):
        # Every other slot unread: volumes are scaled for the gap, ratio stays 1.
        _fill(live_ingestor, {"US": _Tally(articles=10)},
              skip=lambda s: s.minute in (15, 45))
        raw = next(r for r in await _fetch(live_ingestor) if r.entity == "US")
        mag = next(r for r in live_ingestor.normalise(raw) if r.signal_type == "news_magnitude")
        assert mag.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_published_gap_counts_as_zero_not_missing(self, live_ingestor):
        _fill(live_ingestor, {"US": _Tally(articles=10)})
        live_ingestor._slots[LATEST] = None  # GDELT 404'd this slot
        raw = next(r for r in await _fetch(live_ingestor) if r.entity == "US")
        assert raw.raw["coverage_24h"] == 1.0
        assert raw.raw["articles_24h"] == pytest.approx(950)

    @pytest.mark.asyncio
    async def test_mostly_unread_window_skips_cycle(self, live_ingestor):
        _fill(live_ingestor, {"US": _Tally(articles=10)},
              skip=lambda s: s > LATEST - timedelta(hours=18))
        assert await _fetch(live_ingestor) == []

    @pytest.mark.asyncio
    async def test_cold_baseline_omits_magnitude_only(self, live_ingestor):
        _fill(live_ingestor, {"US": _Tally(articles=10, tone_x_articles=-20.0)},
              skip=lambda s: s <= LATEST - timedelta(hours=24))
        raw = next(r for r in await _fetch(live_ingestor) if r.entity == "US")
        types = {r.signal_type for r in live_ingestor.normalise(raw)}
        assert types == {"news_sentiment"}

    @pytest.mark.asyncio
    async def test_every_country_gets_a_raw(self, live_ingestor):
        _fill(live_ingestor, {"US": _Tally(articles=10)})
        assert {r.entity for r in await _fetch(live_ingestor)} == set(COUNTRY_FIPS_MAP)


class TestSyncSlots:
    @pytest.mark.asyncio
    async def test_drops_slots_older_than_history(self, live_ingestor):
        old = LATEST - _HISTORY - _SLOT
        live_ingestor._slots[old] = {}
        with patch.object(live_ingestor, "_load_slot", AsyncMock()):
            await live_ingestor._sync_slots(LATEST)
        assert old not in live_ingestor._slots

    @pytest.mark.asyncio
    async def test_requests_only_missing_slots(self, live_ingestor):
        _fill(live_ingestor, {}, skip=lambda s: s == LATEST)
        with patch.object(live_ingestor, "_load_slot", AsyncMock()) as load:
            await live_ingestor._sync_slots(LATEST)
        assert [c.args[2] for c in load.call_args_list] == [LATEST]


class TestNormalise:
    def _raw(self, **kw):
        d = {
            "articles_24h": 500.0, "events_24h": 120, "avg_tone": -3.0,
            "avg_goldstein": -1.5, "baseline_daily_articles": 250.0,
            "window_end": LATEST.isoformat(), "coverage_24h": 1.0,
            "coverage_baseline": 1.0, **kw,
        }
        return RawRecord(source="gdelt", entity="IR", raw=d, fetched_at=LATEST)

    def test_emits_both_signals(self, ingestor):
        recs = {r.signal_type: r for r in ingestor.normalise(self._raw())}
        assert set(recs) == {"news_sentiment", "news_magnitude"}
        assert recs["news_magnitude"].value == pytest.approx(2 / 3)
        assert recs["news_magnitude"].metadata["ratio"] == pytest.approx(2.0)
        assert all(r.source == "gdelt" and r.entity == "IR" for r in recs.values())

    def test_thin_coverage_suppresses_tone(self, ingestor):
        recs = ingestor.normalise(self._raw(articles_24h=5.0))
        assert [r.signal_type for r in recs] == ["news_magnitude"]

    def test_no_articles_no_tone(self, ingestor):
        recs = ingestor.normalise(self._raw(articles_24h=0.0, avg_tone=None, avg_goldstein=None))
        assert [r.signal_type for r in recs] == ["news_magnitude"]
        assert recs[0].value == 0.0
