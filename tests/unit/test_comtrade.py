"""Unit tests for the Comtrade AI supply-chain ingestor — no network calls.

The fake client answers each preview-API call from fixture rows keyed by the
call's parameters, so fetch() runs end to end against known numbers.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from exo.ingestion import comtrade
from exo.ingestion.comtrade import ComtradeIngestor
from exo.models import RawRecord

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def _row(reporter, partner, cmd, value, year=2025):
    return {"refYear": year, "reporterCode": reporter, "partnerCode": partner, "cmdCode": cmd,
            "customsCode": "C00", "motCode": 0, "partner2Code": 0, "primaryValue": value}


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Client:
    """Serves fixture rows: answer(params) -> list of rows."""

    def __init__(self, answer):
        self._answer = answer
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params):
        self.calls.append(params)
        rows = self._answer(params)
        return _Resp({"count": len(rows), "data": rows})


async def _fetch(answer):
    ing = ComtradeIngestor.__new__(ComtradeIngestor)
    ing._pace = 0
    client = _Client(answer)
    with patch("exo.ingestion.comtrade.httpx.AsyncClient", return_value=client), \
         patch.object(ComtradeIngestor, "utcnow", return_value=NOW), \
         patch("exo.ingestion.comtrade.asyncio.sleep", return_value=None):
        raws = await ing.fetch()
    return ing, {r.entity: r for r in raws}, client


def _signals(ing, raw):
    return {r.signal_type: r for r in ing.normalise(raw)}


def _answer(params):
    """US imports half its hardware from China; Japan none; Russia doesn't report."""
    rep, cmd, flow = params.get("reporterCode"), params["cmdCode"], params["flowCode"]
    if flow == "M" and cmd == ",".join(comtrade.HW_CODES) and params["period"] == 2025:
        if rep == 842:
            return [_row(842, 156, "8542", 500e9), _row(842, 392, "8542", 500e9)]
        if rep == 392:
            return [_row(392, 842, "8471", 10e9)]
        return []
    if flow == "X" and cmd == ",".join(comtrade.HW_CODES):   # mirror: exports to non-reporters
        if rep == 156:
            return [_row(156, 643, "8542", 3e9, 2024)]
        if rep == 276:
            return [_row(276, 643, "8542", 1e9, 2024)]
        return []
    if flow == "M" and cmd == "2605":                         # cobalt: DR Congo 80% of supply
        return [_row(156, 180, "2605", 800e6, 2024), _row(156, 710, "2605", 200e6, 2024)]
    if flow == "M" and cmd == "8101" and params.get("period") == 2022:   # tungsten, all reporters
        return [_row(842, 0, "8101", 100e6, 2022), _row(842, 156, "8101", 60e6, 2022),
                _row(392, 0, "8101", 1e6, 2022), _row(392, 156, "8101", 1e6, 2022)]
    return []


@pytest.mark.asyncio
async def test_hardware_share_shrunk_toward_mean():
    ing, raws, _ = await _fetch(_answer)
    us = _signals(ing, raws["US"])["hw_contested_share"]
    assert us.metadata["raw_share"] == pytest.approx(0.5)
    assert us.metadata["basis"] == "own" and us.metadata["year"] == 2025
    # $1tn basket: shrinkage toward the mean (0.25 over US + JP) is negligible.
    assert us.value == pytest.approx(0.5, abs=1e-3)
    jp = _signals(ing, raws["JP"])["hw_contested_share"]
    # $10bn basket of 0% is pulled up by 1e8/(1e10+1e8) of the mean.
    assert jp.value == pytest.approx(0.25 * 1e8 / (10e9 + 1e8), abs=1e-4)


@pytest.mark.asyncio
async def test_non_reporter_rebuilt_from_mirror():
    ing, raws, _ = await _fetch(_answer)
    ru = _signals(ing, raws["RU"])["hw_contested_share"]
    assert ru.metadata["basis"] == "mirror"
    assert ru.metadata["raw_share"] == pytest.approx(0.75)   # 3bn of 4bn from China


@pytest.mark.asyncio
async def test_chokepoint_is_largest_origin_share():
    ing, raws, _ = await _fetch(_answer)
    cd = _signals(ing, raws["CD"])["supplier_chokepoint"]
    assert cd.metadata["top_hs_code"] == "2605"
    assert cd.metadata["top_share"] == pytest.approx(0.8)
    assert cd.value == 1.0   # 80% is past the 50% saturation
    za = _signals(ing, raws["ZA"])["supplier_chokepoint"]
    assert za.value == pytest.approx(0.2 / comtrade.CHOKEPOINT_SATURATION)


@pytest.mark.asyncio
async def test_cn_mineral_dependence_and_floor():
    ing, raws, _ = await _fetch(_answer)
    us = _signals(ing, raws["US"])["cn_mineral_dependence"]
    assert us.metadata["baseline_year"] == 2022
    assert us.metadata["groups"] == {"tungsten": 0.6}
    # Only one group, so the weight cancels: (60m + w*5m) / (100m + 5m), w = 61/101.
    w = 61e6 / 101e6
    assert us.value == pytest.approx((60e6 + w * 5e6) / 105e6, abs=1e-4)
    # Japan's $1m basket is under the $10m floor: no record.
    assert "cn_mineral_dependence" not in _signals(ing, raws["JP"])
    # China's dependence on itself is not a risk.
    assert _signals(ing, raws["CN"])["cn_mineral_dependence"].value == 0.0


@pytest.mark.asyncio
async def test_no_chokepoint_data_writes_no_zero():
    # Every call fails: nothing is returned, so no country is scored "supplies nothing".
    _, raws, _ = await _fetch(lambda p: [])
    assert raws == {}
    # A run that got hardware but no chokepoint origins carries None, not (0, "").
    ing = ComtradeIngestor.__new__(ComtradeIngestor)
    raw = RawRecord(source="comtrade", entity="CD", fetched_at=NOW,
                    raw={"hardware": None, "chokepoint": None, "minerals": {}})
    assert ing.normalise(raw) == []


def test_codes_cover_tracked_countries():
    from exo.risk_index.engine import COUNTRIES
    assert set(comtrade.TRACKED) == set(COUNTRIES)
    assert comtrade.ISO2_TO_COMTRADE["FR"] == 251   # Comtrade's France, not M49 250
    assert comtrade.ISO2_TO_COMTRADE["IN"] == 699
    assert comtrade.ISO2_TO_COMTRADE["TW"] == 490
    assert {"2805", "8103", "282200"}.isdisjoint(comtrade.CHOKEPOINT_CODES)


@pytest.mark.asyncio
async def test_capped_call_is_split_by_importer():
    def answer(params):
        importers = str(params["reporterCode"]).split(",")
        if len(importers) > 6:   # pretend anything over 6 importers hits the cap
            return [_row(842, 180, "2605", 1.0)] * comtrade._ROW_CAP
        return [_row(int(i), 180, "2605", 1.0) for i in importers]
    ing = ComtradeIngestor.__new__(ComtradeIngestor)
    ing._pace = 0
    client = _Client(answer)
    with patch("exo.ingestion.comtrade.asyncio.sleep", return_value=None):
        rows = await ing._importer_rows(client, comtrade.CHOKEPOINT_IMPORTERS, 2024, "2605")
    assert len(rows) == len(comtrade.CHOKEPOINT_IMPORTERS)   # one per importer, none duplicated
    assert len(client.calls) == 3                            # 12 capped, then 6 + 6
