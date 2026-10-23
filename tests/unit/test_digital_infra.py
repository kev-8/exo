"""Unit tests for the digital infrastructure ingestor — no network calls."""

from collections import Counter
from datetime import datetime, timezone

import pytest

from exo.ingestion.digital_infra import (
    CABLE_COUNTRY_NAMES,
    DigitalInfraIngestor,
    cable_countries,
    p90,
    scarcity,
)
from exo.models import RawRecord
from exo.risk_index.engine import COUNTRIES

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def test_scarcity_scale():
    assert scarcity(0, 17) == 1.0
    assert scarcity(17, 17) == 0.0
    assert scarcity(90, 17) == 0.0            # above the p90 clips to 0
    assert scarcity(4, 17) == pytest.approx(1 - 1.6094 / 2.8904, abs=1e-3)


def test_p90_is_ninetieth_percentile_of_all_countries():
    counts = Counter({f"C{i}": i for i in range(1, 11)})   # 1..10
    assert p90(counts) == 9


def test_planned_and_future_cables_dont_count():
    lp = [{"country": "Guinea"}, {"country": "Guinea"}, {"country": "France"}]
    assert cable_countries({"rfs_year": 2020, "landing_points": lp}, 2026) == {"Guinea", "France"}
    assert cable_countries({"rfs_year": 2028, "landing_points": lp}, 2026) == set()
    assert cable_countries({"rfs_year": 2020, "is_planned": True, "landing_points": lp}, 2026) == set()


def test_normalise_writes_every_tracked_country():
    raw = RawRecord(source="digital_infra", entity="ixp_count", fetched_at=NOW,
                    raw={"counts": {iso2: 0 for iso2 in COUNTRIES} | {"US": 212},
                         "p90": 17, "n_countries": 171})
    recs = {r.entity: r for r in DigitalInfraIngestor.__new__(DigitalInfraIngestor).normalise(raw)}
    assert set(recs) == set(COUNTRIES)
    assert recs["US"].value == 0.0 and recs["KP"].value == 1.0
    assert recs["US"].metadata == {"raw_count": 212, "p90": 17, "n_countries": 171}


def test_cable_names_cover_tracked_countries():
    assert set(CABLE_COUNTRY_NAMES) == set(COUNTRIES)
