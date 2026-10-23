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
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
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


async def _fetch_series(
    client: httpx.AsyncClient,
    reporter: str,
    partners: list[str],
    indicator: str,
    since_year: int,
) -> dict[str, dict[int, float]]:
    """One SDMX request: {partner: {year: USD}} for the partners that have data.

    "Total" is WITS's all-products code. The service answers in XML unless
    JSON is asked for in the Accept header (a format= parameter is ignored).
    """
    url = f"{WITS_BASE}/A.{reporter}.{'+'.join(partners)}.Total.{indicator}/"
    resp = await client.get(
        url,
        params={"startPeriod": since_year},
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:  # "NoRecordsFound"
        return {}
    resp.raise_for_status()
    body = resp.json()

    dims = body["structure"]["dimensions"]
    series_dims = dims["series"]
    partner_pos = next(i for i, d in enumerate(series_dims) if d["id"] == "PARTNER")
    partner_ids = [v["id"] for v in series_dims[partner_pos]["values"]]
    years = [int(v["id"]) for v in dims["observation"][0]["values"]]

    out: dict[str, dict[int, float]] = {}
    for key, series in body["dataSets"][0]["series"].items():
        partner = partner_ids[int(key.split(":")[partner_pos])]
        out[partner] = {
            years[int(i)]: float(obs[0]) * 1000.0   # WITS reports thousands of USD
            for i, obs in series.get("observations", {}).items()
            if obs and obs[0] is not None
        }
    return out


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


class WITSIngestor(BaseIngestor):
    """Ingest trade concentration and secondary sanctions exposure from WITS."""

    source = "wits"

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=90.0) as client:
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

        return records
