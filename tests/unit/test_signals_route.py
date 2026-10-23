"""The signal feed serves every country-level source, one latest record per signal."""

from datetime import datetime, timedelta, timezone

import pytest

from api.routes import signals
from exo.models import FeatureRecord
from exo.store.feature_store import FeatureStore

NOW = datetime.now(timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = FeatureStore(data_dir=tmp_path / "features", redis_url=None)
    monkeypatch.setattr(signals, "_store", s)
    return s


def _write(store, source, signal_type, value, age_days=0, entity="JP"):
    store.write(FeatureRecord(source=source, entity=entity, signal_type=signal_type, value=value,
                              metadata={}, as_of_ts=NOW - timedelta(days=age_days)))


def test_new_sources_served_latest_once_newest_first(store):
    _write(store, "world_bank", "electricity_access", 99.0, age_days=3)
    _write(store, "world_bank", "electricity_access", 100.0, age_days=1)
    _write(store, "export_controls", "entity_list_count", 0.25)
    _write(store, "ucdp_ged", "ucdp_ged_events", 0.0, age_days=2)
    _write(store, "fred", "economic_indicator", 0.3)            # US only: not served for JP
    _write(store, "export_controls", "entity_list_count", 0.9, entity="CN")
    out = signals.get_latest_signals("jp", limit=40)
    assert [(s.source, s.signal_type, s.value) for s in out] == [
        ("export_controls", "entity_list_count", 0.25),
        ("world_bank", "electricity_access", 100.0),
        ("ucdp_ged", "ucdp_ged_events", 0.0),
    ]


def test_every_feed_source_has_a_ui_badge():
    from pathlib import Path
    feed = (Path(__file__).parents[2] / "ui/src/components/SignalFeed.jsx").read_text()
    for source in signals.FEED_SIGNALS:
        assert f"  {source}:" in feed, f"no SOURCE_STYLE badge for {source}"
