"""WITS bilateral trade ingestor — weekly.

Fetches bilateral trade flows from the World Integrated Trade Solution (WITS)
public REST API and emits two signals per country:

  trade_concentration  — (trade_with_US + trade_with_EU) / total_trade
                         Higher = more dependent on western trading partners
                         = higher sanctions vulnerability. A reporter's own
                         bloc is left out (an EU member's trade with the EU,
                         the US's with itself): sanctions leverage comes from
                         outside a country's bloc, not from its single market.

  secondary_exposure   — trade_with_comprehensively_sanctioned / total_trade,
                         scaled so SECONDARY_SATURATION of trade scores 1.0.
                         Exposure to secondary sanctions for dealing with the
                         countries under broad OFAC programmes (SANCTIONED_PARTNERS).

  top_export_partners  — the 5 largest export markets, each with its value,
                         share of exports and top 3 HS goods groups. Served
                         by /api/trade; the value is those 5 partners' share.

Data is annual; weekly polling picks up new WITS releases.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

WITS_BASE = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/DF_WITS_TradeStats_Trade"

# ISO 3166-1 alpha-2 → WITS reporter/partner code
ISO2_TO_WITS: dict[str, str] = {
    "US": "USA", "RU": "RUS", "CN": "CHN", "UA": "UKR",
    "IL": "ISR", "IR": "IRN", "IN": "IND", "PK": "PAK",
    "KP": "PRK", "TW": "TWN",
    "HT": "HTI", "BR": "BRA", "MX": "MEX", "NG": "NGA",
    "KE": "KEN", "ZA": "ZAF", "FR": "FRA", "GB": "GBR",
    "MY": "MYS", "CL": "CHL",
    "KR": "KOR", "JP": "JPN", "DE": "DEU", "NL": "NLD",
    "AE": "ARE", "SG": "SGP", "VN": "VNM", "ID": "IDN",
    "CD": "ZAR", "GN": "GIN",  # WITS still codes DR Congo as ZAR, not COD
    # Extended set — potential sanctioned partners (not tracked countries)
    "BY": "BLR", "VE": "VEN", "SY": "SYR", "CU": "CUB",
    "SD": "SDN", "MM": "MMR", "IQ": "IRQ", "LB": "LBN",
    "LY": "LBY", "SO": "SOM", "YE": "YEM", "ZW": "ZWE",
}

# Reverse mapping: WITS code → ISO2 (for SDN lookup)
WITS_TO_ISO2: dict[str, str] = {v: k for k, v in ISO2_TO_WITS.items()}

# Tracked reporter countries (subset of ISO2_TO_WITS)
TRACKED_COUNTRIES: dict[str, str] = {
    "US": "USA", "RU": "RUS", "CN": "CHN", "UA": "UKR",
    "IL": "ISR", "IR": "IRN", "IN": "IND", "PK": "PAK",
    "KP": "PRK", "TW": "TWN",
    "HT": "HTI", "BR": "BRA", "MX": "MEX", "NG": "NGA",
    "KE": "KEN", "ZA": "ZAF", "FR": "FRA", "GB": "GBR",
    "MY": "MYS", "CL": "CHL",
    "KR": "KOR", "JP": "JPN", "DE": "DEU", "NL": "NLD",
    "AE": "ARE", "SG": "SGP", "VN": "VNM", "ID": "IDN",
    "CD": "ZAR", "GN": "GIN",
}

PARTNER_WORLD = "WLD"
PARTNER_USA = "USA"
# WITS lists the EU (EUN) as a partner but holds no trade rows for it, so EU
# trade is the sum over member states.
EU_MEMBERS = (
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
    "MLT", "NLD", "POL", "PRT", "ROM", "SVK", "SVN", "ESP", "SWE",   # WITS codes Romania ROM
)

INDICATORS = ["XPRT-TRD-VL", "MPRT-TRD-VL"]  # export value, import value

# Countries under broad OFAC country programmes, whose trading partners face
# secondary-sanctions risk. Chosen by programme, not by SDN entry count: SDN
# counts also reflect entities registered in hubs (UAE, UK) and cartel or
# terrorism designations (Mexico), which aren't sanctions on the country.
SANCTIONED_PARTNERS: tuple[str, ...] = ("RUS", "IRN", "PRK", "CUB", "SYR", "BLR", "VEN")

# Share of total trade with SANCTIONED_PARTNERS that scores secondary_exposure 1.0.
# Tracked countries top out near 7% (India, on Russian oil); 10% leaves room
# above that while still separating them from the near-zero Western shares.
SECONDARY_SATURATION = 0.10

# Years searched back from now for the newest year the reporter has data for.
_LOOKBACK_YEARS = 6
# The service rejects keys listing more than ~45 partners with a 400.
_MAX_PARTNERS = 40


async def _get_series(
    client: httpx.AsyncClient,
    key: str,
    since_year: int,
    until_year: int | None = None,
) -> list[tuple[dict[str, str], dict[int, float]]]:
    """One SDMX request: [(dimension codes, {year: USD})] per series with data.

    The service answers in XML unless JSON is asked for in the Accept header
    (a format= parameter is ignored); a key position left empty matches every
    code; very long keys are rejected as invalid URLs.
    """
    params = {"startPeriod": since_year}
    if until_year is not None:
        params["endPeriod"] = until_year
    resp = await client.get(f"{WITS_BASE}/{key}/", params=params,
                            headers={"Accept": "application/json"})
    if resp.status_code == 404:  # "NoRecordsFound"
        return []
    resp.raise_for_status()
    body = resp.json()

    dims = body["structure"]["dimensions"]
    series_dims = dims["series"]
    years = [int(v["id"]) for v in dims["observation"][0]["values"]]
    out = []
    for key_idx, series in body["dataSets"][0]["series"].items():
        codes = {d["id"]: d["values"][int(i)]["id"] for d, i in zip(series_dims, key_idx.split(":"))}
        out.append((codes, {
            years[int(i)]: float(obs[0]) * 1000.0   # WITS reports thousands of USD
            for i, obs in series.get("observations", {}).items()
            if obs and obs[0] is not None
        }))
    return out


async def _fetch_series(
    client: httpx.AsyncClient,
    reporter: str,
    partners: list[str],
    indicator: str,
    since_year: int,
) -> dict[str, dict[int, float]]:
    """{partner: {year: USD}} for all-products ("Total") trade with *partners*."""
    rows = await _get_series(client, f"A.{reporter}.{'+'.join(partners)}.Total.{indicator}", since_year)
    return {codes["PARTNER"]: obs for codes, obs in rows}


async def _fetch_partner_values(
    client: httpx.AsyncClient,
    reporter: str,
    partners: list[str],
    indicator: str,
    since_year: int,
) -> tuple[int, dict[str, float]] | None:
    """Fetch one indicator for many partners, _MAX_PARTNERS per request.

    Returns (year, {partner: USD}) for the newest year the reporter has a
    World total, with every partner read at that same year so shares aren't
    mixed across years. Partners with no flow that year are absent. None when
    the reporter has no World total or a request fails.
    """
    partners = [p for p in dict.fromkeys(partners) if p != PARTNER_WORLD]
    by_partner: dict[str, dict[int, float]] = {}
    try:
        for i in range(0, len(partners), _MAX_PARTNERS):
            chunk = [PARTNER_WORLD, *partners[i:i + _MAX_PARTNERS]]
            by_partner.update(await _fetch_series(client, reporter, chunk, indicator, since_year))
    except Exception as exc:
        logger.warning("WITS fetch failed reporter=%s indicator=%s: %s", reporter, indicator, exc)
        return None

    world = by_partner.get(PARTNER_WORLD)
    if not world:
        return None
    year = max(world)
    return year, {p: v[year] for p, v in by_partner.items() if year in v}


# ---------------------------------------------------------------------------
# Top export partners (served by /api/trade)
# ---------------------------------------------------------------------------

TOP_PARTNERS = 5
WB_COUNTRIES_URL = "https://api.worldbank.org/v2/country"

# WITS partner codes that differ from the World Bank's ISO3 codes.
WITS_PARTNER_OVERRIDES: dict[str, tuple[str, str]] = {
    "ZAR": ("CD", "DR Congo"), "ROM": ("RO", "Romania"), "OAS": ("TW", "Taiwan"),
    "SER": ("RS", "Serbia"), "MNT": ("ME", "Montenegro"), "TMP": ("TL", "Timor-Leste"),
    "SUD": ("SD", "Sudan"),
}
# World Bank names shortened for display.
NAME_OVERRIDES: dict[str, str] = {
    "Korea, Rep.": "South Korea", "Korea, Dem. People's Rep.": "North Korea",
    "Hong Kong SAR, China": "Hong Kong", "Macao SAR, China": "Macao",
    "Russian Federation": "Russia", "Viet Nam": "Vietnam", "Iran, Islamic Rep.": "Iran",
    "Egypt, Arab Rep.": "Egypt", "Turkiye": "Turkey", "Congo, Dem. Rep.": "DR Congo",
    "Congo, Rep.": "Congo", "Venezuela, RB": "Venezuela", "Yemen, Rep.": "Yemen",
    "Lao PDR": "Laos", "Kyrgyz Republic": "Kyrgyzstan", "Slovak Republic": "Slovakia",
    "Czechia": "Czech Republic", "Syrian Arab Republic": "Syria", "Gambia, The": "Gambia",
    "Bahamas, The": "Bahamas", "Micronesia, Fed. Sts.": "Micronesia",
    "United Arab Emirates": "UAE", "United Kingdom": "UK",
}

# HS section groups in WITS, with display labels. Requested in two halves:
# all 16 in one key makes the URL too long for the service.
HS_GROUPS: dict[str, str] = {
    "01-05_Animal": "Animal Products", "06-15_Vegetable": "Vegetables & Grains",
    "16-24_FoodProd": "Food & Beverages", "25-26_Minerals": "Ores & Minerals",
    "27-27_Fuels": "Fuels", "28-38_Chemicals": "Chemicals",
    "39-40_PlastiRub": "Plastics & Rubber", "41-43_HidesSkin": "Hides & Leather",
    "44-49_Wood": "Wood & Paper", "50-63_TextCloth": "Textiles & Clothing",
    "64-67_Footwear": "Footwear", "68-71_StoneGlas": "Stone, Glass & Gold",
    "72-83_Metals": "Metals", "84-85_MachElec": "Machinery & Electronics",
    "86-89_Transport": "Vehicles & Transport", "90-99_Miscellan": "Miscellaneous",
}
GOODS_PER_PARTNER = 3


async def _country_lookup(client: httpx.AsyncClient) -> dict[str, tuple[str, str]]:
    """WITS partner code → (ISO2, display name), countries only (no aggregates)."""
    resp = await client.get(WB_COUNTRIES_URL, params={"format": "json", "per_page": 400})
    resp.raise_for_status()
    lookup = {
        c["id"]: (c["iso2Code"], NAME_OVERRIDES.get(c["name"], c["name"]))
        for c in resp.json()[1]
        if c["region"]["value"] != "Aggregates"
    }
    return {**lookup, **WITS_PARTNER_OVERRIDES}


async def _top_export_partners(
    client: httpx.AsyncClient,
    reporter: str,
    year: int,
    lookup: dict[str, tuple[str, str]],
) -> list[dict] | None:
    """The reporter's TOP_PARTNERS largest export markets in *year*, with goods."""
    rows = await _get_series(client, f"A.{reporter}..Total.XPRT-TRD-VL", year, year)
    exports = {codes["PARTNER"]: obs.get(year, 0.0) for codes, obs in rows}
    world = exports.get(PARTNER_WORLD, 0.0)
    if world <= 0:
        return None
    top = sorted(
        ((v, p) for p, v in exports.items() if p in lookup and p != reporter and v > 0),
        reverse=True,
    )[:TOP_PARTNERS]
    if not top:
        return None

    codes = [p for _, p in top]
    goods: dict[str, dict[str, float]] = {p: {} for p in codes}
    groups = list(HS_GROUPS)
    half = len(groups) // 2
    for chunk in (groups[:half], groups[half:]):
        for c, obs in await _get_series(
            client, f"A.{reporter}.{'+'.join(codes)}.{'+'.join(chunk)}.XPRT-TRD-VL", year, year,
        ):
            goods[c["PARTNER"]][c["PRODUCTCODE"]] = obs.get(year, 0.0)

    return [
        {
            "iso2": lookup[p][0],
            "name": lookup[p][1],
            "trade_usd_b": round(v / 1e9, 1),
            "share_pct": round(100 * v / world, 1),
            "goods": [HS_GROUPS[g] for g, _ in sorted(goods[p].items(), key=lambda kv: -kv[1])
                      if goods[p][g] > 0][:GOODS_PER_PARTNER],
        }
        for v, p in top
    ]


class WITSIngestor(BaseIngestor):
    """Ingest trade concentration and secondary sanctions exposure from WITS."""

    source = "wits"

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                lookup = await _country_lookup(client)
            except Exception as exc:
                logger.warning("World Bank country list failed; skipping top export partners: %s", exc)
                lookup = None

            for iso2, wits_code in TRACKED_COUNTRIES.items():
                others = [p for p in SANCTIONED_PARTNERS if p != wits_code]
                partners = [PARTNER_WORLD, PARTNER_USA, *EU_MEMBERS, *others]
                world_total = us_total = eu_total = 0.0
                sanctioned_trade: dict[str, float] = {}
                data_years: dict[str, int] = {}

                for indicator in INDICATORS:
                    got = await _fetch_partner_values(
                        client, wits_code, partners, indicator, now.year - _LOOKBACK_YEARS,
                    )
                    if got is None:
                        continue
                    year, values = got
                    data_years[indicator] = year
                    world_total += values[PARTNER_WORLD]
                    us_total += values.get(PARTNER_USA, 0.0)
                    eu_total += sum(values.get(m, 0.0) for m in EU_MEMBERS if m != wits_code)
                    for p in others:
                        if values.get(p, 0.0) > 0:
                            sanctioned_trade[p] = sanctioned_trade.get(p, 0.0) + values[p]

                if world_total <= 0:
                    logger.info("WITS: no usable trade data for %s", iso2)
                    continue

                partners = None
                if lookup and "XPRT-TRD-VL" in data_years:
                    try:
                        partners = await _top_export_partners(
                            client, wits_code, data_years["XPRT-TRD-VL"], lookup)
                    except Exception as exc:
                        logger.warning("WITS top export partners failed for %s: %s", iso2, exc)

                raws.append(RawRecord(
                    source=self.source,
                    entity=iso2,
                    raw={
                        "world_total": world_total,
                        "us_total": us_total,
                        "eu_total": eu_total,
                        "sanctioned_total": sum(sanctioned_trade.values()),
                        "sanctioned_by_partner": sanctioned_trade,
                        "data_year": max(data_years.values()),
                        "top_export_partners": partners,
                        "export_year": data_years.get("XPRT-TRD-VL"),
                    },
                    fetched_at=now,
                ))

        logger.info("WITS: trade data for %d of %d countries", len(raws), len(TRACKED_COUNTRIES))
        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw
        records: list[FeatureRecord] = []

        world_total: float = d["world_total"]
        if world_total <= 0:
            return []

        # Signal 1: trade_concentration (US/EU dependency, own bloc excluded)
        wits_code = TRACKED_COUNTRIES.get(raw.entity)
        us_pct = 0.0 if wits_code == PARTNER_USA else d.get("us_total", 0.0) / world_total
        eu_pct = 0.0 if wits_code in EU_MEMBERS else d.get("eu_total", 0.0) / world_total
        concentration = min(1.0, us_pct + eu_pct)
        records.append(FeatureRecord(
            source=self.source,
            entity=raw.entity,
            signal_type="trade_concentration",
            value=round(concentration, 4),
            metadata={
                "us_pct": round(us_pct, 4),
                "eu_pct": round(eu_pct, 4),
                "world_total_usd": world_total,
                "data_year": d.get("data_year"),
            },
            as_of_ts=now,
        ))

        # Signal 2: secondary_exposure (trade share with sanctioned countries).
        # Zero trade is a real 0, not missing data, so it's always emitted.
        sanctioned_total: float = d.get("sanctioned_total", 0.0)
        share = sanctioned_total / world_total
        records.append(FeatureRecord(
            source=self.source,
            entity=raw.entity,
            signal_type="secondary_exposure",
            value=round(min(1.0, share / SECONDARY_SATURATION), 4),
            metadata={
                "sanctioned_usd": sanctioned_total,
                "sanctioned_pct": round(share, 4),
                "world_total_usd": world_total,
                "by_partner_usd": d.get("sanctioned_by_partner", {}),
                "data_year": d.get("data_year"),
            },
            as_of_ts=now,
        ))

        # Signal 3: top export partners for the trade panel. The value is the
        # share of exports going to those partners (export concentration).
        partners = d.get("top_export_partners")
        if partners:
            records.append(FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="top_export_partners",
                value=round(min(1.0, sum(p["share_pct"] for p in partners) / 100), 4),
                metadata={"partners": partners, "year": d.get("export_year")},
                as_of_ts=now,
            ))

        return records
