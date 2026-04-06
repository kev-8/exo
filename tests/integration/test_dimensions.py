"""Integration tests for dimension scorers — seeds feature store with signals."""

from datetime import datetime, timezone

import pytest

from exo.models import FeatureRecord
from exo.risk_index.dimensions import DimensionScorer
from exo.store.feature_store import FeatureStore


@pytest.fixture
def store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def scorer(store):
    return DimensionScorer(store=store)


def _now():
    return datetime.now(timezone.utc)


def _write(store, source, entity, signal_type, value, metadata=None):
    store.write(FeatureRecord(
        source=source,
        entity=entity,
        signal_type=signal_type,
        value=value,
        metadata=metadata or {},
        as_of_ts=_now(),
    ))


# ---------------------------------------------------------------------------
# 1. Political stability — WGI
# ---------------------------------------------------------------------------

class TestPoliticalStability:
    def test_defaults_to_0_5_with_no_data(self, scorer):
        dim = scorer.political_stability("XX", _now())
        assert dim.score == 0.5
        assert dim.name == "political_stability"
        assert dim.weight == 0.25

    def test_wgi_political_stability_inverted(self, scorer, store):
        # WGI=1.0 (well-governed) → instability = 1 - 1.0 = 0.0
        _write(store, "world_bank", "US", "political_stability", 1.0)
        dim = scorer.political_stability("US", _now())
        assert abs(dim.score - 0.0) < 1e-6

    def test_wgi_voice_accountability_inverted(self, scorer, store):
        _write(store, "world_bank", "RU", "voice_accountability", 0.2)
        dim = scorer.political_stability("RU", _now())
        assert abs(dim.score - 0.8) < 1e-6  # 1 - 0.2

    def test_two_wgi_signals_averaged(self, scorer, store):
        _write(store, "world_bank", "CN", "political_stability", 0.6)
        _write(store, "world_bank", "CN", "voice_accountability", 0.4)
        dim = scorer.political_stability("CN", _now())
        # instability = (1-0.6 + 1-0.4) / 2 = (0.4 + 0.6) / 2 = 0.5
        assert abs(dim.score - 0.5) < 1e-6

    def test_contributing_signals_present(self, scorer, store):
        _write(store, "world_bank", "IL", "political_stability", 0.3)
        dim = scorer.political_stability("IL", _now())
        assert any("wgi_political_stability" in s for s in dim.contributing_signals)


# ---------------------------------------------------------------------------
# 2. Conflict intensity — GDELT magnitude
# ---------------------------------------------------------------------------

class TestConflictIntensity:
    def test_defaults_to_0_3_with_no_data(self, scorer):
        dim = scorer.conflict_intensity("XX", _now())
        assert abs(dim.score - 0.3) < 1e-6

    def test_gdelt_magnitude_zero_not_added(self, scorer, store):
        # magnitude=0.0 must not be appended (it would mask the 0.3 default)
        _write(store, "gdelt", "US", "news_magnitude", 0.0)
        dim = scorer.conflict_intensity("US", _now())
        assert abs(dim.score - 0.3) < 1e-6

    def test_gdelt_magnitude_nonzero_used(self, scorer, store):
        _write(store, "gdelt", "UA", "news_magnitude", 0.8)
        dim = scorer.conflict_intensity("UA", _now())
        assert abs(dim.score - 0.8) < 1e-6

    def test_weight(self, scorer):
        dim = scorer.conflict_intensity("XX", _now())
        assert dim.weight == 0.25


# ---------------------------------------------------------------------------
# 3. Policy predictability — UNGA ideal point variance
# ---------------------------------------------------------------------------

class TestPolicyPredictability:
    def test_defaults_to_0_5_with_no_data(self, scorer):
        dim = scorer.policy_predictability("XX", _now())
        assert abs(dim.score - 0.5) < 1e-6
        assert "insufficient_data" in dim.contributing_signals

    def test_uses_unga_signal(self, scorer, store):
        _write(store, "unga_votes", "RU", "ideal_point_variance", 0.592,
               metadata={"variance": 0.148, "n_years": 10, "latest_year": 2024})
        dim = scorer.policy_predictability("RU", _now())
        assert abs(dim.score - 0.592) < 1e-4

    def test_low_variance_low_risk(self, scorer, store):
        _write(store, "unga_votes", "UA", "ideal_point_variance", 0.022,
               metadata={"variance": 0.0055, "n_years": 10, "latest_year": 2024})
        dim = scorer.policy_predictability("UA", _now())
        assert dim.score < 0.1

    def test_weight(self, scorer):
        dim = scorer.policy_predictability("XX", _now())
        assert dim.weight == 0.20

    def test_contributing_signals_include_metadata(self, scorer, store):
        _write(store, "unga_votes", "IN", "ideal_point_variance", 0.046,
               metadata={"variance": 0.0115, "n_years": 8, "latest_year": 2023})
        dim = scorer.policy_predictability("IN", _now())
        assert any("ideal_point_variance" in s for s in dim.contributing_signals)
        assert any("n_years=8" in s for s in dim.contributing_signals)


# ---------------------------------------------------------------------------
# 4. Sanctions risk — GDELT sentiment + Google Trends
# ---------------------------------------------------------------------------

class TestSanctionsRisk:
    def test_defaults_to_0_3_with_no_data(self, scorer):
        dim = scorer.sanctions_risk("XX", _now())
        assert abs(dim.score - 0.3) < 1e-6

    def test_negative_gdelt_tone_raises_risk(self, scorer, store):
        # sentiment=-0.4 → sanctions_score = (1 - (-0.4)) / 2 = 0.7
        _write(store, "gdelt", "IR", "news_sentiment", -0.4)
        dim = scorer.sanctions_risk("IR", _now())
        assert dim.score > 0.5

    def test_positive_gdelt_tone_lowers_risk(self, scorer, store):
        _write(store, "gdelt", "US", "news_sentiment", 0.5)
        dim = scorer.sanctions_risk("US", _now())
        assert dim.score < 0.4

    def test_google_trends_combined(self, scorer, store):
        _write(store, "gdelt", "RU", "news_sentiment", -0.3)
        _write(store, "google_trends", "sanctions", "search_volume", 0.8)
        dim = scorer.sanctions_risk("RU", _now())
        # sanctions_score(gdelt) = (1 - (-0.3)) / 2 = 0.65
        # combined = (0.65 + 0.8) / 2 = 0.725
        assert abs(dim.score - 0.725) < 1e-4

    def test_weight(self, scorer):
        dim = scorer.sanctions_risk("XX", _now())
        assert dim.weight == 0.15


# ---------------------------------------------------------------------------
# 5. Economic stress — FRED + World Bank + EIA + Finnhub
# ---------------------------------------------------------------------------

class TestEconomicStress:
    def test_defaults_to_0_5_with_no_data(self, scorer):
        dim = scorer.economic_stress("XX", _now())
        assert abs(dim.score - 0.5) < 1e-6

    def test_fred_signal_used(self, scorer, store):
        _write(store, "fred", "US", "economic_indicator", 0.7)
        dim = scorer.economic_stress("US", _now())
        assert dim.score > 0.5

    def test_eia_crude_importer_high_price_raises_stress(self, scorer, store):
        # energy_import_weight=1.0 (full importer), crude=$120 → stress=1.0
        _write(store, "world_bank", "UA", "energy_imports_pct", 1.0)
        _write(store, "eia", "WTI", "crude_oil_price", 120.0)
        dim = scorer.economic_stress("UA", _now())
        assert dim.score > 0.5

    def test_eia_crude_exporter_high_price_lowers_stress(self, scorer, store):
        # energy_import_weight=0.0 (full exporter), crude=$120 → directional = 0.0
        _write(store, "world_bank", "RU", "energy_imports_pct", 0.0)
        _write(store, "eia", "WTI", "crude_oil_price", 120.0)
        dim = scorer.economic_stress("RU", _now())
        assert dim.score < 0.5

    def test_finnhub_negative_news_raises_stress(self, scorer, store):
        # sentiment=-1.0 → stress = (1 - (-1)) / 2 = 1.0
        _write(store, "finnhub", "general", "news_sentiment", -1.0)
        dim = scorer.economic_stress("US", _now())
        assert dim.score > 0.5

    def test_wgi_rule_of_law_inverted(self, scorer, store):
        # rule_of_law=1.0 (good governance) → normalised = 1 - 1.0 = 0.0 (low stress)
        _write(store, "world_bank", "US", "rule_of_law", 1.0)
        dim = scorer.economic_stress("US", _now())
        assert dim.score < 0.5

    def test_weight(self, scorer):
        dim = scorer.economic_stress("XX", _now())
        assert dim.weight == 0.15


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestCompositeWeights:
    def test_weights_sum_to_one(self, scorer):
        dims = scorer.score_all("XX", _now())
        assert abs(sum(d.weight for d in dims) - 1.0) < 1e-6

    def test_all_five_dimensions_returned(self, scorer):
        dims = scorer.score_all("XX", _now())
        names = {d.name for d in dims}
        assert names == {
            "political_stability", "conflict_intensity", "policy_predictability",
            "sanctions_risk", "economic_stress",
        }

    def test_all_scores_in_range(self, scorer, store):
        _write(store, "world_bank", "IL", "political_stability", 0.4)
        _write(store, "gdelt", "IL", "news_magnitude", 0.9)
        _write(store, "gdelt", "IL", "news_sentiment", -0.35)
        _write(store, "unga_votes", "IL", "ideal_point_variance", 0.44)
        _write(store, "eia", "WTI", "crude_oil_price", 75.0)
        dims = scorer.score_all("IL", _now())
        for d in dims:
            assert 0.0 <= d.score <= 1.0, f"{d.name}: {d.score}"
