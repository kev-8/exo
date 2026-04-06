"""Unit tests for data model schema validation."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from exo.models import (
    DimensionScore,
    FeatureRecord,
    FeatureQuery,
    FeatureUpdated,
    KalshiMarket,
    PipelineEvent,
    RawRecord,
    RiskIndexSnapshot,
    StalenessAlert,
)


class TestFeatureRecord:
    def test_required_fields(self):
        now = datetime.now(timezone.utc)
        rec = FeatureRecord(
            source="gdelt",
            entity="US",
            signal_type="news_sentiment",
            value=0.5,
            metadata={"key": "val"},
            as_of_ts=now,
        )
        assert rec.source == "gdelt"
        assert rec.value == 0.5
        assert isinstance(rec.record_id, str)
        assert UUID(rec.record_id)

    def test_ticker_optional(self):
        rec = FeatureRecord(
            source="fred",
            entity="US",
            signal_type="economic_indicator",
            value=0.3,
            metadata={},
            as_of_ts=datetime.now(timezone.utc),
        )
        assert rec.ticker is None

    def test_ingested_at_auto(self):
        rec = FeatureRecord(
            source="x", entity="y", signal_type="z",
            value=0.0, metadata={}, as_of_ts=datetime.now(timezone.utc),
        )
        assert rec.ingested_at is not None


class TestRiskIndexSnapshot:
    def test_composite_score_range(self):
        dims = [
            DimensionScore("political_stability", 0.4, 0.25, ["s1"]),
            DimensionScore("conflict_intensity", 0.6, 0.25, ["s2"]),
            DimensionScore("policy_predictability", 0.5, 0.20, ["s3"]),
            DimensionScore("sanctions_risk", 0.3, 0.15, ["s4"]),
            DimensionScore("economic_stress", 0.5, 0.15, ["s5"]),
        ]
        composite = sum(d.score * d.weight for d in dims)
        snap = RiskIndexSnapshot(
            country="US",
            composite_score=composite,
            dimensions=dims,
            as_of_ts=datetime.now(timezone.utc),
        )
        assert 0 <= snap.composite_score <= 1

    def test_snapshot_id_unique(self):
        now = datetime.now(timezone.utc)
        dims = [DimensionScore("political_stability", 0.5, 1.0, [])]
        s1 = RiskIndexSnapshot(country="US", composite_score=0.5, dimensions=dims, as_of_ts=now)
        s2 = RiskIndexSnapshot(country="US", composite_score=0.5, dimensions=dims, as_of_ts=now)
        assert s1.snapshot_id != s2.snapshot_id


class TestPipelineEvents:
    def test_feature_updated_has_id(self):
        event = FeatureUpdated()
        assert UUID(event.event_id)

    def test_staleness_alert_fields(self):
        alert = StalenessAlert(
            source="gdelt",
            entity="US",
            threshold_sec=900,
            message="stale",
        )
        assert alert.source == "gdelt"

    def test_event_id_unique(self):
        e1 = FeatureUpdated()
        e2 = FeatureUpdated()
        assert e1.event_id != e2.event_id


class TestKalshiMarket:
    def test_spread(self):
        market = KalshiMarket(
            ticker="TEST",
            title="test",
            category="test",
            status="open",
            yes_ask=0.52,
            yes_bid=0.48,
            no_ask=0.52,
            no_bid=0.48,
            volume=10_000,
            open_interest=500,
            close_time=None,
        )
        assert market.yes_ask - market.yes_bid == pytest.approx(0.04)
