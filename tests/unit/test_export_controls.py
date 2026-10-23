"""Unit tests for the export-control list ingestor — no network calls."""

from datetime import datetime, timezone

import pytest

from exo.ingestion.export_controls import (
    CSL_ENTITY_LIST,
    ExportControlsIngestor,
    _country_counts,
    parse_csl,
    parse_opensanctions,
)
from exo.models import RawRecord
from exo.risk_index.engine import COUNTRIES

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)

CSV_HEADER = '"id","schema","name","countries","program_ids","sanctions"\n'


def test_csl_keeps_entity_list_only_and_one_count_per_country():
    body = {"results": [
        {"source": CSL_ENTITY_LIST, "addresses": [{"country": "CN"}, {"country": "CN"}, {"country": "hk"}]},
        {"source": CSL_ENTITY_LIST, "addresses": [{"country": "RU"}]},
        {"source": "Specially Designated Nationals (SDN) - Treasury Department",
         "addresses": [{"country": "IR"}]},
    ]}
    entries = parse_csl(body)
    assert entries == [{"CN", "HK"}, {"RU"}]
    # Hong Kong folds into China, and an entity is counted once per country.
    assert _country_counts(entries) == {"CN": 1, "RU": 1}


def test_opensanctions_program_filter_and_missing_countries():
    text = CSV_HEADER + (
        '"a","Company","Lockheed","us","CN-UEL","UEL"\n'
        '"b","Company","Recorded Future","","CN-UEL","UEL"\n'
        '"c","Company","Northrop","us","CN-CML","AFSL"\n'      # counter-sanctions, not ECL/UEL
        '"d","Company","Airbus Defence","de;fr","CN-ECL","ECL"\n'
    )
    entries, no_country = parse_opensanctions(text, ("CN-ECL", "CN-UEL"))
    assert len(entries) == 3 and no_country == 1
    assert _country_counts(entries) == {"US": 1, "DE": 1, "FR": 1}


def test_normalise_log_scale_and_real_zeros():
    counts = {iso2: 0 for iso2 in COUNTRIES} | {"CN": 1000, "RU": 99}
    raw = RawRecord(source="export_controls", entity="entity_list_count", fetched_at=NOW,
                    raw={"counts": counts, "list_max": 1000, "list_total": 1200, "no_country": 0})
    recs = {r.entity: r for r in ExportControlsIngestor.__new__(ExportControlsIngestor).normalise(raw)}
    assert set(recs) == set(COUNTRIES)
    assert recs["CN"].value == 1.0
    assert recs["RU"].value == pytest.approx(0.6667, abs=1e-4)   # log(100) / log(1001)
    assert recs["JP"].value == 0.0 and recs["JP"].metadata["raw_count"] == 0
    assert all(r.signal_type == "entity_list_count" for r in recs.values())
