"""The AI exposure rollback trigger flags week-on-week jumps over the threshold."""

from datetime import datetime, timedelta, timezone

from exo.models import DimensionScore, RiskIndexSnapshot
from exo.risk_index.validation import ai_exposure_jumps
from exo.store.index_store import IndexStore

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def _snap(store, country, ai_score, ts, with_ai=True):
    dims = [DimensionScore("economic_stress", 0.5, 0.9, [])]
    if with_ai:
        dims.append(DimensionScore("ai_exposure", ai_score, 0.1, [f"ai={ai_score}"]))
    store.write(RiskIndexSnapshot(country=country, composite_score=0.5, dimensions=dims, as_of_ts=ts))


def test_flags_only_jumps_over_threshold(tmp_path):
    store = IndexStore(data_dir=tmp_path / "idx")
    week_ago = NOW - timedelta(days=8)
    _snap(store, "CN", 0.60, week_ago); _snap(store, "CN", 0.70, NOW)   # +0.10: flagged
    _snap(store, "JP", 0.33, week_ago); _snap(store, "JP", 0.36, NOW)   # +0.03: fine
    _snap(store, "KR", 0.0, week_ago, with_ai=False); _snap(store, "KR", 0.9, NOW)  # no baseline
    jumps = ai_exposure_jumps(store, ["CN", "JP", "KR", "US"], now=NOW)
    assert set(jumps) == {"CN"}
    assert jumps["CN"]["before"] == 0.60 and jumps["CN"]["after"] == 0.70
    assert jumps["CN"]["signals_after"] == ["ai=0.7"]
