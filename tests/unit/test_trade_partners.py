"""Live trade panel: WITS top export partners and the /api/trade fallback."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from exo.ingestion import wits
from exo.models import FeatureRecord, RawRecord

LOOKUP = {"USA": ("US", "United States"), "CHN": ("CN", "China"), "OAS": ("TW", "Taiwan"),
          "KOR": ("KR", "South Korea"), "VNM": ("VN", "Vietnam"), "DEU": ("DE", "Germany"),
          "JPN": ("JP", "Japan")}


def _body(series):
    """SDMX-JSON with one observation (2023) per series; series: [(partner, product, thousands)]."""
    partners = sorted({p for p, _, _ in series})
    products = sorted({g for _, g, _ in series})
    return {
        "structure": {"dimensions": {
            "series": [
                {"id": "FREQ", "values": [{"id": "A"}]},
                {"id": "REPORTER", "values": [{"id": "JPN"}]},
                {"id": "PARTNER", "values": [{"id": p} for p in partners]},
                {"id": "PRODUCTCODE", "values": [{"id": g} for g in products]},
                {"id": "INDICATOR", "values": [{"id": "XPRT-TRD-VL"}]},
            ],
            "observation": [{"id": "TIME_PERIOD", "values": [{"id": "2023"}]}],
        }},
        "dataSets": [{"series": {
            f"0:0:{partners.index(p)}:{products.index(g)}:0": {"observations": {"0": [v, 0]}}
            for p, g, v in series
        }}],
    }


def _client(bodies):
    client = AsyncMock()
    resps = []
    for b in bodies:
        r = MagicMock(status_code=200)
        r.json.return_value = b
        resps.append(r)
    client.get.side_effect = resps
    return client


@pytest.mark.asyncio
async def test_top_partners_skip_aggregates_and_self_with_goods():
    bn = 1e6   # WITS reports thousands of USD
    totals = _body([("WLD", "Total", 1000 * bn), ("EAS", "Total", 600 * bn),   # aggregates
                    ("JPN", "Total", 999 * bn),                                # self
                    ("USA", "Total", 200 * bn), ("CHN", "Total", 180 * bn), ("OAS", "Total", 60 * bn),
                    ("KOR", "Total", 50 * bn), ("VNM", "Total", 30 * bn), ("DEU", "Total", 20 * bn)])
    half1 = _body([("USA", "27-27_Fuels", 1.0), ("CHN", "28-38_Chemicals", 40.0)])
    half2 = _body([("USA", "86-89_Transport", 90.0), ("USA", "84-85_MachElec", 60.0),
                   ("CHN", "84-85_MachElec", 70.0)])
    out = await wits._top_export_partners(_client([totals, half1, half2]), "JPN", 2023, LOOKUP)
    assert [p["iso2"] for p in out] == ["US", "CN", "TW", "KR", "VN"]   # top 5, Germany dropped
    assert out[0] == {"iso2": "US", "name": "United States", "trade_usd_b": 200.0, "share_pct": 20.0,
                      "goods": ["Vehicles & Transport", "Machinery & Electronics", "Fuels"]}
    assert out[1]["goods"] == ["Machinery & Electronics", "Chemicals"]
    assert out[2]["goods"] == []


def test_normalise_emits_partners_record():
    partners = [{"iso2": "US", "name": "United States", "trade_usd_b": 145.1, "share_pct": 20.2, "goods": []},
                {"iso2": "CN", "name": "China", "trade_usd_b": 126.4, "share_pct": 17.6, "goods": []}]
    raw = RawRecord(source="wits", entity="JP", fetched_at=datetime.now(timezone.utc), raw={
        "world_total": 1e12, "us_total": 0.1e12, "eu_total": 0.1e12, "sanctioned_total": 0.0,
        "top_export_partners": partners, "export_year": 2023})
    rec = next(r for r in wits.WITSIngestor.__new__(wits.WITSIngestor).normalise(raw)
               if r.signal_type == "top_export_partners")
    assert rec.value == pytest.approx(0.378)
    assert rec.metadata == {"partners": partners, "year": 2023}


def test_romania_uses_the_wits_code():
    assert "ROM" in wits.EU_MEMBERS and "ROU" not in wits.EU_MEMBERS


class TestTradeRoute:
    @pytest.fixture
    def route(self, tmp_path, monkeypatch):
        from api.routes import trade
        from exo.store.feature_store import FeatureStore
        store = FeatureStore(data_dir=tmp_path / "features", redis_url=None)
        monkeypatch.setattr(trade, "_store", store)
        return trade, store

    def test_prefers_live_wits_record(self, route):
        trade, store = route
        store.write(FeatureRecord(source="wits", entity="JP", signal_type="top_export_partners", value=0.4,
                                  as_of_ts=datetime.now(timezone.utc), metadata={"year": 2023, "partners": [
                                      {"iso2": "US", "name": "United States", "trade_usd_b": 145.1,
                                       "share_pct": 20.2, "goods": ["Vehicles & Transport"]}]}))
        out = trade.get_trade_flows("jp")
        assert out.source == "wits" and out.year == 2023 and out.partners[0].iso2 == "US"

    def test_falls_back_to_static_file(self, route):
        trade, _ = route
        out = trade.get_trade_flows("US")          # in trade_flows.json, no live record
        assert out.source == "static" and out.partners

    def test_no_data_is_404(self, route):
        from fastapi import HTTPException
        trade, _ = route
        with pytest.raises(HTTPException) as exc:
            trade.get_trade_flows("GN")            # not in the static file, no live record
        assert exc.value.status_code == 404
