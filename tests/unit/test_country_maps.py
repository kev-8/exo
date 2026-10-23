"""Every source's country map must cover every tracked country.

Each source keys countries by its own code system (FIPS for GDELT,
Gleditsch-Ward for UCDP, ZAR for DR Congo in WITS), so a country added to
COUNTRIES but missed in one map loses that signal silently.
"""

import pytest

from api.routes.countries import COUNTRY_META
from exo.ingestion import gdelt, ucdp, unga_votes, wgi, wits, world_bank
from exo.risk_index.engine import COUNTRIES

MAPS = {
    "api COUNTRY_META": set(COUNTRY_META),
    "world_bank COUNTRIES": set(world_bank.COUNTRIES),
    "wits TRACKED_COUNTRIES": set(wits.TRACKED_COUNTRIES),
    "wgi ISO2_TO_ISO3": set(wgi.ISO2_TO_ISO3),
    "unga_votes ISO3_TO_ISO2": set(unga_votes.ISO3_TO_ISO2.values()),
    "ucdp ISO2_TO_UCDP": set(ucdp.ISO2_TO_UCDP),
    "gdelt COUNTRY_FIPS_MAP": set(gdelt.COUNTRY_FIPS_MAP),
}


@pytest.mark.parametrize("name", MAPS)
def test_map_covers_tracked_countries(name):
    assert set(COUNTRIES) - MAPS[name] == set()


def test_no_duplicate_tracked_countries():
    assert len(COUNTRIES) == len(set(COUNTRIES)) == 30


def test_codes_that_differ_from_iso():
    assert gdelt.COUNTRY_FIPS_MAP["SG"] == ("SN",)   # SG is Senegal in FIPS
    assert gdelt.COUNTRY_FIPS_MAP["CD"] == ("CG",)
    assert wits.TRACKED_COUNTRIES["CD"] == "ZAR"
    assert ucdp.ISO2_TO_UCDP["CD"] == 490
