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
        # (1.0 - (-0.40)) / 2 = 0.70
        assert abs(records[0].value - 0.70) < 1e-6

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


# ---------------------------------------------------------------------------
# World Bank — trade openness (new derived signal)
# ---------------------------------------------------------------------------

class TestWorldBankTradeOpenness:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.world_bank import WorldBankIngestor
        return WorldBankIngestor.__new__(WorldBankIngestor)

    def test_trade_openness_emitted(self, ingestor):
        raw = _raw("world_bank", "DE", {"NE.EXP.GNFS.ZS": 50.0, "NE.IMP.GNFS.ZS": 40.0})
        records = ingestor.normalise(raw)
        assert any(r.signal_type == "trade_openness" for r in records)

    def test_trade_openness_value(self, ingestor):
        # (50 + 40) / 200 = 0.45
        raw = _raw("world_bank", "DE", {"NE.EXP.GNFS.ZS": 50.0, "NE.IMP.GNFS.ZS": 40.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "trade_openness")
        assert abs(rec.value - 0.45) < 1e-6

    def test_trade_openness_capped_at_one(self, ingestor):
        # Singapore-like: (175 + 160) / 200 = 1.675 → capped at 1.0
        raw = _raw("world_bank", "SG", {"NE.EXP.GNFS.ZS": 175.0, "NE.IMP.GNFS.ZS": 160.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "trade_openness")
        assert rec.value == 1.0

    def test_trade_openness_metadata(self, ingestor):
        raw = _raw("world_bank", "US", {"NE.EXP.GNFS.ZS": 12.0, "NE.IMP.GNFS.ZS": 14.0})
        records = ingestor.normalise(raw)
        rec = next(r for r in records if r.signal_type == "trade_openness")
        assert rec.metadata["exports_pct_gdp"] == 12.0
        assert rec.metadata["imports_pct_gdp"] == 14.0

    def test_trade_openness_not_emitted_when_missing(self, ingestor):
        # Only exports available — cannot compute openness
        raw = _raw("world_bank", "US", {"NE.EXP.GNFS.ZS": 12.0})
        records = ingestor.normalise(raw)
        assert all(r.signal_type != "trade_openness" for r in records)


# ---------------------------------------------------------------------------
# WITS — bilateral trade concentration
# ---------------------------------------------------------------------------

class TestWITSNormalise:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.wits import WITSIngestor
        return WITSIngestor.__new__(WITSIngestor)

    def test_concentration_value(self, ingestor):
        # us=300B, eu=200B, world=1T → concentration = 0.5
        raw = _raw("wits", "RU", {
            "world_total": 1_000_000_000_000.0,
            "us_total":      300_000_000_000.0,
            "eu_total":      200_000_000_000.0,
        })
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert abs(records[0].value - 0.5) < 1e-6

    def test_concentration_capped_at_one(self, ingestor):
        raw = _raw("wits", "CA", {
            "world_total": 1_000.0,
            "us_total":    900.0,
            "eu_total":    900.0,  # sum > world due to re-exports
        })
        records = ingestor.normalise(raw)
        assert records[0].value == 1.0

    def test_metadata_contains_us_eu_pct(self, ingestor):
        raw = _raw("wits", "MX", {
            "world_total": 1_000.0,
            "us_total":    750.0,
            "eu_total":    100.0,
        })
        records = ingestor.normalise(raw)
        meta = records[0].metadata
        assert abs(meta["us_pct"] - 0.75) < 1e-6
        assert abs(meta["eu_pct"] - 0.10) < 1e-6

    def test_zero_world_total_returns_empty(self, ingestor):
        raw = _raw("wits", "KP", {"world_total": 0.0, "us_total": 0.0, "eu_total": 0.0})
        assert ingestor.normalise(raw) == []

    def test_signal_type_and_source(self, ingestor):
        raw = _raw("wits", "CN", {
            "world_total": 5e12, "us_total": 0.8e12, "eu_total": 0.5e12,
        })
        records = ingestor.normalise(raw)
        assert records[0].signal_type == "trade_concentration"
        assert records[0].source == "wits"


# ---------------------------------------------------------------------------
# UCDP — GED and Candidate ingestors
# ---------------------------------------------------------------------------

class TestUCDPGEDNormalise:
    @pytest.fixture
    def ingestor(self, tmp_path):
        from exo.ingestion.ucdp import UCDPGEDIngestor
        from exo.store.feature_store import FeatureStore
        inst = UCDPGEDIngestor.__new__(UCDPGEDIngestor)
        inst.store = FeatureStore(data_dir=tmp_path / "features", redis_url=None)
        return inst

    def test_events_and_fatalities_emitted(self, ingestor):
        raw = _raw("ucdp_ged", "UA", {"event_count": 200, "total_fatalities": 5000})
        records = ingestor.normalise(raw)
        signal_types = {r.signal_type for r in records}
        assert "ucdp_ged_events" in signal_types
        assert "ucdp_ged_fatalities" in signal_types

    def test_normalised_to_0_1(self, ingestor):
        raw = _raw("ucdp_ged", "UA", {"event_count": 100, "total_fatalities": 1000})
        records = ingestor.normalise(raw)
        for r in records:
            assert 0.0 <= r.value <= 1.0

    def test_zero_events_zero_score(self, ingestor):
        raw = _raw("ucdp_ged", "TW", {"event_count": 0, "total_fatalities": 0})
        records = ingestor.normalise(raw)
        for r in records:
            assert r.value == 0.0

    def test_rolling_max_in_metadata(self, ingestor):
        raw = _raw("ucdp_ged", "SY", {"event_count": 300, "total_fatalities": 8000})
        records = ingestor.normalise(raw)
        events_rec = next(r for r in records if r.signal_type == "ucdp_ged_events")
        assert "rolling_max" in events_rec.metadata

    def test_source(self, ingestor):
        raw = _raw("ucdp_ged", "IR", {"event_count": 50, "total_fatalities": 200})
        records = ingestor.normalise(raw)
        assert all(r.source == "ucdp_ged" for r in records)


class TestOFACNormalise:
    @pytest.fixture
    def ingestor(self, tmp_path):
        from exo.ingestion.ofac import OFACIngestor
        from exo.store.feature_store import FeatureStore
        inst = OFACIngestor.__new__(OFACIngestor)
        inst.store = FeatureStore(data_dir=tmp_path / "features", redis_url=None)
        return inst

    def test_sdn_count_emitted(self, ingestor):
        raw = _raw("ofac", "RU", {"sdn_count": 3200})
        records = ingestor.normalise(raw)
        assert len(records) == 1
        assert records[0].signal_type == "sdn_entity_count"

    def test_normalised_to_0_1(self, ingestor):
        raw = _raw("ofac", "IR", {"sdn_count": 1500})
        records = ingestor.normalise(raw)
        assert 0.0 <= records[0].value <= 1.0

    def test_zero_count_zero_score(self, ingestor):
        raw = _raw("ofac", "TW", {"sdn_count": 0})
        records = ingestor.normalise(raw)
        assert records[0].value == 0.0

    def test_raw_count_in_metadata(self, ingestor):
        raw = _raw("ofac", "KP", {"sdn_count": 250})
        records = ingestor.normalise(raw)
        assert records[0].metadata["raw_count"] == 250

    def test_rolling_max_in_metadata(self, ingestor):
        raw = _raw("ofac", "VE", {"sdn_count": 400})
        records = ingestor.normalise(raw)
        assert "rolling_max" in records[0].metadata

    def test_source(self, ingestor):
        raw = _raw("ofac", "SY", {"sdn_count": 100})
        records = ingestor.normalise(raw)
        assert records[0].source == "ofac"

    def test_higher_count_higher_score(self, ingestor):
        raw_low = _raw("ofac", "CU", {"sdn_count": 50})
        raw_high = _raw("ofac", "RU", {"sdn_count": 3000})
        score_low = ingestor.normalise(raw_low)[0].value
        score_high = ingestor.normalise(raw_high)[0].value
        assert score_high > score_low


class TestWITSSecondaryExposure:
    @pytest.fixture
    def ingestor(self):
        from exo.ingestion.wits import WITSIngestor
        return WITSIngestor.__new__(WITSIngestor)

    def test_secondary_exposure_emitted_when_present(self, ingestor):
        raw = _raw("wits", "CN", {
            "world_total": 5e12,
            "us_total": 0.8e12,
            "eu_total": 0.5e12,
            "sanctioned_weighted": 0.3e12,
            "n_sanctioned_partners": 2,
        })
        records = ingestor.normalise(raw)
        signal_types = {r.signal_type for r in records}
        assert "trade_concentration" in signal_types
        assert "secondary_exposure" in signal_types

    def test_secondary_exposure_not_emitted_when_zero(self, ingestor):
        raw = _raw("wits", "US", {
            "world_total": 5e12,
            "us_total": 0.0,
            "eu_total": 0.5e12,
            "sanctioned_weighted": 0.0,
            "n_sanctioned_partners": 0,
        })
        records = ingestor.normalise(raw)
        assert all(r.signal_type != "secondary_exposure" for r in records)

    def test_secondary_exposure_value(self, ingestor):
        # sanctioned_weighted = 0.5e12 * 0.8 (sdn_score baked in)
        # world = 5e12 → secondary = 0.4e12 / 5e12 = 0.08
        raw = _raw("wits", "IN", {
            "world_total": 5e12,
            "us_total": 1e12,
            "eu_total": 0.5e12,
            "sanctioned_weighted": 0.4e12,
            "n_sanctioned_partners": 3,
        })
        records = ingestor.normalise(raw)
        sec = next(r for r in records if r.signal_type == "secondary_exposure")
        assert abs(sec.value - 0.08) < 1e-6

    def test_secondary_exposure_capped_at_one(self, ingestor):
        raw = _raw("wits", "RU", {
            "world_total": 1e12,
            "us_total": 0.0,
            "eu_total": 0.0,
            "sanctioned_weighted": 2e12,  # exceeds world total
            "n_sanctioned_partners": 5,
        })
        records = ingestor.normalise(raw)
        sec = next(r for r in records if r.signal_type == "secondary_exposure")
        assert sec.value == 1.0

    def test_secondary_exposure_metadata(self, ingestor):
        raw = _raw("wits", "PK", {
            "world_total": 1e12,
            "us_total": 0.1e12,
            "eu_total": 0.1e12,
            "sanctioned_weighted": 0.2e12,
            "n_sanctioned_partners": 2,
        })
        records = ingestor.normalise(raw)
        sec = next(r for r in records if r.signal_type == "secondary_exposure")
        assert sec.metadata["n_sanctioned_partners"] == 2


class TestUCDPCandidateNormalise:
    @pytest.fixture
    def ingestor(self, tmp_path):
        from exo.ingestion.ucdp import UCDPCandidateIngestor
        from exo.store.feature_store import FeatureStore
        inst = UCDPCandidateIngestor.__new__(UCDPCandidateIngestor)
        inst.store = FeatureStore(data_dir=tmp_path / "features", redis_url=None)
        return inst

    def test_only_events_emitted(self, ingestor):
        # Candidate dataset omits fatalities
        raw = _raw("ucdp_candidate", "RU", {"event_count": 80, "total_fatalities": 0})
        records = ingestor.normalise(raw)
        signal_types = {r.signal_type for r in records}
        assert "ucdp_candidate_events" in signal_types
        assert "ucdp_candidate_fatalities" not in signal_types

    def test_normalised_to_0_1(self, ingestor):
        raw = _raw("ucdp_candidate", "IN", {"event_count": 30, "total_fatalities": 0})
        records = ingestor.normalise(raw)
        for r in records:
            assert 0.0 <= r.value <= 1.0

    def test_source(self, ingestor):
        raw = _raw("ucdp_candidate", "PK", {"event_count": 20, "total_fatalities": 0})
        records = ingestor.normalise(raw)
        assert all(r.source == "ucdp_candidate" for r in records)
