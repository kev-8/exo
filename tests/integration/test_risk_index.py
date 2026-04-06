"""Integration tests for the risk index engine."""

import tempfile
from datetime import datetime, timezone

import pytest

from exo.models import FeatureRecord
from exo.risk_index.engine import RiskIndexEngine
from exo.store.feature_store import FeatureStore
from exo.store.index_store import IndexStore


@pytest.fixture
def tmp_store(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features", redis_url=None)


@pytest.fixture
def tmp_index(tmp_path):
    return IndexStore(data_dir=tmp_path / "risk_index")


class TestRiskIndexIsolation:
    def test_update_writes_to_index_store_only(self, tmp_store, tmp_index):
        """RiskIndexEngine.update() writes to the index store, not the feature store."""
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)

        engine.update("US", as_of_ts=now)

        snaps = tmp_index.get_history("US")
        assert len(snaps) == 1
        assert snaps[0].country == "US"
        assert 0 <= snaps[0].composite_score <= 1

        from exo.models import FeatureQuery
        fs_recs = tmp_store.read(FeatureQuery(source="risk_index"))
        assert len(fs_recs) == 0

    def test_update_returns_none(self, tmp_store, tmp_index):
        """RiskIndexEngine.update() must return None."""
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        result = engine.update("CN")
        assert result is None

    def test_all_five_dimensions_present(self, tmp_store, tmp_index):
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        engine.update("RU")
        snap = tmp_index.get_latest("RU")
        assert snap is not None
        dimension_names = {d.name for d in snap.dimensions}
        assert dimension_names == {
            "political_stability",
            "conflict_intensity",
            "policy_predictability",
            "sanctions_risk",
            "economic_stress",
        }

    def test_dimension_scores_in_range(self, tmp_store, tmp_index):
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        engine.update("IN")
        snap = tmp_index.get_latest("IN")
        assert snap is not None
        for dim in snap.dimensions:
            assert 0.0 <= dim.score <= 1.0, f"{dim.name}: {dim.score}"

    def test_weights_sum_to_one(self, tmp_store, tmp_index):
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        engine.update("US")
        snap = tmp_index.get_latest("US")
        assert abs(sum(d.weight for d in snap.dimensions) - 1.0) < 1e-6

    def test_update_all(self, tmp_store, tmp_index):
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        engine.update_all(countries=["US", "CN"])
        for country in ["US", "CN"]:
            assert tmp_index.get_latest(country) is not None

    def test_history_accumulates(self, tmp_store, tmp_index):
        engine = RiskIndexEngine(store=tmp_store, index_store=tmp_index)
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
        engine.update("UA", as_of_ts=t1)
        engine.update("UA", as_of_ts=t2)
        assert len(tmp_index.get_history("UA")) == 2
