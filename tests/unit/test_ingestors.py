"""Unit tests for ingestor normalise() logic — no network calls."""

from datetime import datetime, timezone

import pytest

from exo.models import RawRecord


def _now():
    return datetime.now(timezone.utc)


def _raw(source, entity, raw):
    return RawRecord(source=source, entity=entity, raw=raw, fetched_at=_now())


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

class TestGDELTNormalise:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.gdelt import GDELTIngestor
        return GDELTIngestor.__new__(GDELTIngestor)  # no store/bus needed for normalise

    def test_sentiment_record_emitted(self, ingestor):
        raw = _raw("gdelt", "RU", {
            "tone_score": -0.35,
            "keyword": "Russia", "start_date": "2026-04-04", "end_date": "2026-04-05",
        })
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert records[0].signal_type == "news_sentiment"

    def test_sentiment_value(self, ingestor):
        raw = _raw("gdelt", "IL", {
            "tone_score": -0.40,
            "keyword": "Israel", "start_date": "2026-04-04", "end_date": "2026-04-05",
        })
        records = ingestor.normalise(raw)
        assert abs(records[0].value - (-0.40)) < 1e-6

    def test_missing_tone_returns_empty(self, ingestor):
        raw = _raw("gdelt", "CN", {
            "tone_score": None,
            "keyword": "China", "start_date": "2026-04-04", "end_date": "2026-04-05",
        })
        assert ingestor.normalise(raw) == []

    def test_source_and_entity(self, ingestor):
        raw = _raw("gdelt", "UA", {
            "tone_score": -0.25,
            "keyword": "Ukraine", "start_date": "2026-04-04", "end_date": "2026-04-05",
        })
        records = ingestor.normalise(raw)
        assert records[0].source == "gdelt"
        assert records[0].entity == "UA"


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

class TestPolymarketNormalise:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.polymarket import PolymarketIngestor
        return PolymarketIngestor.__new__(PolymarketIngestor)

    def test_outcome_prices_as_json_string(self, ingestor):
        raw = _raw("polymarket", "market-1", {
            "outcomePrices": '["0.65", "0.35"]',
            "endDateIso": "2027-01-01",
            "question": "Test?",
            "event_title": "Test Event",
            "tag": "politics",
        })
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert abs(records[0].value - 0.65) < 1e-6

    def test_outcome_prices_as_list(self, ingestor):
        raw = _raw("polymarket", "market-2", {
            "outcomePrices": [0.40, 0.60],
            "endDateIso": "2027-01-01",
            "question": "Test?",
            "event_title": "Test Event",
            "tag": "world",
        })
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert abs(records[0].value - 0.40) < 1e-6

    def test_missing_outcome_prices_returns_empty(self, ingestor):
        raw = _raw("polymarket", "market-3", {
            "endDateIso": "2027-01-01",
            "question": "Test?",
        })
        assert ingestor.normalise(raw) == []

    def test_near_term_flag_true(self, ingestor):
        from datetime import timedelta
        soon = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = _raw("polymarket", "market-4", {
            "outcomePrices": '["0.5", "0.5"]',
            "endDateIso": soon,
            "question": "Test?",
            "event_title": "",
            "tag": "politics",
        })
        records = ingestor.normalise(raw)
        assert records[0].metadata["near_term"] is True

    def test_near_term_flag_false(self, ingestor):
        raw = _raw("polymarket", "market-5", {
            "outcomePrices": '["0.3", "0.7"]',
            "endDateIso": "2028-12-31",
            "question": "Test?",
            "event_title": "",
            "tag": "elections",
        })
        records = ingestor.normalise(raw)
        assert records[0].metadata["near_term"] is False


# ---------------------------------------------------------------------------
# World Bank
# ---------------------------------------------------------------------------

class TestWorldBankNormalise:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.world_bank import WorldBankIngestor
        return WorldBankIngestor.__new__(WorldBankIngestor)

    def _make_raw(self, data, entity="US"):
        r = RawRecord(source="world_bank", entity=entity, raw=data, fetched_at=_now())
        # WorldBankIngestor.normalise needs self.source
        return r

    def test_wgi_normalisation_midpoint(self, ingestor):
        raw = self._make_raw({"PV.EST": 0.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "political_stability")
        assert abs(rec.value - 0.5) < 1e-6  # (0 + 2.5) / 5

    def test_wgi_normalisation_max(self, ingestor):
        raw = self._make_raw({"PV.EST": 2.5})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "political_stability")
        assert abs(rec.value - 1.0) < 1e-6

    def test_wgi_normalisation_min(self, ingestor):
        raw = self._make_raw({"VA.EST": -2.5})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "voice_accountability")
        assert abs(rec.value - 0.0) < 1e-6

    def test_energy_imports_neutral(self, ingestor):
        raw = self._make_raw({"EG.IMP.CONS.ZS": 0.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "energy_imports_pct")
        assert abs(rec.value - 0.5) < 1e-6  # (0 + 100) / 200

    def test_energy_imports_full_importer(self, ingestor):
        raw = self._make_raw({"EG.IMP.CONS.ZS": 100.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "energy_imports_pct")
        assert abs(rec.value - 1.0) < 1e-6

    def test_energy_imports_full_exporter(self, ingestor):
        raw = self._make_raw({"EG.IMP.CONS.ZS": -100.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "energy_imports_pct")
        assert abs(rec.value - 0.0) < 1e-6

    def test_regular_indicator_stored_as_is(self, ingestor):
        raw = self._make_raw({"NY.GDP.MKTP.KD.ZG": 3.2})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "gdp_growth")
        assert abs(rec.value - 3.2) < 1e-6

    def test_missing_value_skipped(self, ingestor):
        raw = self._make_raw({"NY.GDP.MKTP.KD.ZG": None})
        records = ingestor.normalise(raw)
        assert all(r.signal_type != "gdp_growth" for r in records)


# ---------------------------------------------------------------------------
# UNGA Votes
# ---------------------------------------------------------------------------

class TestUNGAVotesNormalise:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.unga_votes import UNGAVotesIngestor
        return UNGAVotesIngestor.__new__(UNGAVotesIngestor)

    def test_zero_variance_zero_risk(self, ingestor):
        # All ideal points identical → variance=0 → risk=0
        raw = _raw("unga_votes", "US", {
            "ideal_points": [0.5] * 10,
            "latest_year": 2024,
            "latest_ideal_point": 0.5,
            "n_years": 10,
        })
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert records[0].value == 0.0

    def test_high_variance_capped_at_one(self, ingestor):
        # Alternating extremes → high variance, capped at 1.0
        raw = _raw("unga_votes", "KP", {
            "ideal_points": [-3.0, 3.0] * 5,
            "latest_year": 2024,
            "latest_ideal_point": 3.0,
            "n_years": 10,
        })
        records = ingestor.normalise(raw)
        assert records[0].value == 1.0

    def test_known_variance(self, ingestor):
        # variance of [0, 1] = 0.25 → risk = 0.25 / 0.25 = 1.0
        raw = _raw("unga_votes", "RU", {
            "ideal_points": [0.0, 1.0],
            "latest_year": 2024,
            "latest_ideal_point": 1.0,
            "n_years": 2,
        })
        records = ingestor.normalise(raw)
        assert abs(records[0].value - 1.0) < 1e-4

    def test_insufficient_data_returns_empty(self, ingestor):
        raw = _raw("unga_votes", "TW", {
            "ideal_points": [0.5],
            "latest_year": 2024,
            "latest_ideal_point": 0.5,
            "n_years": 1,
        })
        assert ingestor.normalise(raw) == []

    def test_signal_type(self, ingestor):
        raw = _raw("unga_votes", "CN", {
            "ideal_points": [0.3, 0.4, 0.35, 0.38, 0.32],
            "latest_year": 2024,
            "latest_ideal_point": 0.32,
            "n_years": 5,
        })
        records = ingestor.normalise(raw)
        assert records[0].signal_type == "ideal_point_variance"
        assert records[0].source == "unga_votes"

    def test_metadata_present(self, ingestor):
        raw = _raw("unga_votes", "IN", {
            "ideal_points": [0.1, 0.2, 0.15],
            "latest_year": 2023,
            "latest_ideal_point": 0.15,
            "n_years": 3,
        })
        records = ingestor.normalise(raw)
        meta = records[0].metadata
        assert "variance" in meta
        assert meta["latest_year"] == 2023
        assert meta["n_years"] == 3
