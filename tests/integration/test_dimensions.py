"""Integration tests for dimension scorers — seeds feature store with signals."""

from datetime import datetime, timezone

import pytest

from exo.models import FeatureRecord, TierScore
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

    def test_no_signals_defaults_to_0_3(self, scorer, store):
        # With no ACLED or GDELT magnitude data, score stays at 0.3
        dim = scorer.conflict_intensity("US", _now())
        assert abs(dim.score - 0.3) < 1e-6

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
        # ingestor normalises: raw tone=-0.4 → stored value=(1-(-0.4))/2=0.7
        # acute=0.7; structural default=0.3; short_term default=0.3
        # blended = 0.4*0.3 + 0.2*0.3 + 0.4*0.7 = 0.12+0.06+0.28 = 0.46
        _write(store, "gdelt", "IR", "news_sentiment", 0.7)
        dim = scorer.sanctions_risk("IR", _now())
        assert dim.score > 0.3  # elevated above structural-only default

    def test_positive_gdelt_tone_lowers_risk(self, scorer, store):
        # ingestor normalises: raw tone=0.5 → stored value=(1-0.5)/2=0.25
        # acute=0.25; structural default=0.3; short_term default=0.3
        # blended = 0.4*0.3 + 0.2*0.3 + 0.4*0.25 = 0.12+0.06+0.10 = 0.28
        _write(store, "gdelt", "US", "news_sentiment", 0.25)
        dim = scorer.sanctions_risk("US", _now())
        assert dim.score < 0.4

    def test_google_trends_combined(self, scorer, store):
        # ingestor normalises: raw tone=-0.3 → stored value=0.65; google_trends→0.8
        # acute_avg=(0.65+0.8)/2=0.725; structural=0.3, short_term=0.3
        # blended = 0.4*0.3 + 0.2*0.3 + 0.4*0.725 = 0.12+0.06+0.29 = 0.47
        _write(store, "gdelt", "RU", "news_sentiment", 0.65)
        _write(store, "google_trends", "sanctions", "search_volume", 0.8)
        dim = scorer.sanctions_risk("RU", _now())
        assert abs(dim.score - 0.47) < 1e-4

    def test_structural_signals_included(self, scorer, store):
        # trade_openness=0.8, trade_concentration=0.6 → structural avg=0.7
        # short_term=0.3, acute=0.3
        # blended = 0.4*0.7 + 0.2*0.3 + 0.4*0.3 = 0.28+0.06+0.12 = 0.46
        _write(store, "world_bank", "CN", "trade_openness", 0.8,
               metadata={"exports_pct_gdp": 20.0, "imports_pct_gdp": 140.0})
        _write(store, "wits", "CN", "trade_concentration", 0.6,
               metadata={"us_pct": 0.4, "eu_pct": 0.2, "world_total_usd": 5e12})
        dim = scorer.sanctions_risk("CN", _now())
        assert dim.tier_scores["structural"].score > 0.5
        assert abs(dim.score - 0.46) < 1e-4

    def test_tier_scores_populated(self, scorer, store):
        # ingestor normalises: raw tone=-0.2 → stored value=0.6
        _write(store, "gdelt", "IL", "news_sentiment", 0.6)
        dim = scorer.sanctions_risk("IL", _now())
        assert set(dim.tier_scores.keys()) == {"structural", "short_term", "acute"}
        assert all(isinstance(ts, TierScore) for ts in dim.tier_scores.values())

    def test_sdn_entity_count_in_short_term_tier(self, scorer, store):
        _write(store, "ofac", "RU", "sdn_entity_count", 0.85,
               metadata={"raw_count": 3200, "rolling_max": 3200.0})
        dim = scorer.sanctions_risk("RU", _now())
        assert dim.tier_scores["short_term"].score == 0.85
        assert any("sdn_count" in s for s in dim.tier_scores["short_term"].contributing_signals)

    def test_secondary_exposure_in_structural_tier(self, scorer, store):
        _write(store, "wits", "CN", "secondary_exposure", 0.35,
               metadata={"sanctioned_weighted_usd": 1.5e11, "world_total_usd": 4e12, "n_sanctioned_partners": 3})
        dim = scorer.sanctions_risk("CN", _now())
        assert any("secondary_exposure" in s for s in dim.tier_scores["structural"].contributing_signals)

    def test_all_signals_combined(self, scorer, store):
        # structural: openness=0.6, concentration=0.5, secondary=0.4 → avg=0.5
        # short_term: sdn=0.7
        # acute: stored value=0.6 (= ingestor-normalised from raw tone=-0.2)
        # blended = 0.4*0.5 + 0.2*0.7 + 0.4*0.6 = 0.20+0.14+0.24 = 0.58
        _write(store, "world_bank", "IR", "trade_openness", 0.6,
               metadata={"exports_pct_gdp": 50.0, "imports_pct_gdp": 70.0})
        _write(store, "wits", "IR", "trade_concentration", 0.5,
               metadata={"us_pct": 0.2, "eu_pct": 0.3, "world_total_usd": 1e11})
        _write(store, "wits", "IR", "secondary_exposure", 0.4,
               metadata={"sanctioned_weighted_usd": 4e10, "world_total_usd": 1e11, "n_sanctioned_partners": 2})
        _write(store, "ofac", "IR", "sdn_entity_count", 0.7,
               metadata={"raw_count": 1500, "rolling_max": 3200.0})
        _write(store, "gdelt", "IR", "news_sentiment", 0.6)
        dim = scorer.sanctions_risk("IR", _now())
        assert abs(dim.score - 0.58) < 1e-4

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
        # ingestor normalises: raw sentiment=-1.0 → stored value=(1-(-1.0))/2=1.0
        _write(store, "finnhub", "general", "news_sentiment", 1.0)
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
        # ingestor normalises: raw tone=-0.35 → stored value=(1-(-0.35))/2=0.675
        _write(store, "gdelt", "IL", "news_sentiment", 0.675)
        _write(store, "unga_votes", "IL", "ideal_point_variance", 0.44)
        _write(store, "eia", "WTI", "crude_oil_price", 75.0)
        dims = scorer.score_all("IL", _now())
        for d in dims:
            assert 0.0 <= d.score <= 1.0, f"{d.name}: {d.score}"
