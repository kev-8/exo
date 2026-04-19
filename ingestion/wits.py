"""WITS bilateral trade ingestor — weekly.

Fetches bilateral trade flows from the World Integrated Trade Solution (WITS)
public REST API and emits two signals per country:

  trade_concentration  — (trade_with_US + trade_with_EU) / total_trade
                         Higher = more dependent on western trading partners
                         = higher sanctions vulnerability.

  secondary_exposure   — sum(trade_with_partner * sdn_score[partner]) / total_trade
                         Weighted exposure to countries under active US sanctions.
                         Partner set is derived dynamically from OFAC SDN data
                         stored in the feature store — no hardcoded country list.

Data is annual; weekly polling picks up new WITS releases.
OFAC ingestor should run before WITS each cycle so SDN data is fresh.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureQuery, FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

WITS_BASE = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/DF_WITS_TradeStats_Tariff"

# ISO 3166-1 alpha-2 → WITS reporter/partner code
ISO2_TO_WITS: dict[str, str] = {
    "US": "USA", "RU": "RUS", "CN": "CHN", "UA": "UKR",
    "IL": "ISR", "IR": "IRN", "IN": "IND", "PK": "PAK",
    "KP": "PRK", "TW": "TWN",
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
}

PARTNER_WORLD = "WLD"
PARTNER_USA = "USA"
PARTNER_EU = "EUN"   # EU as a bloc in WITS

INDICATORS = ["XPRT-TRD-VL", "MPRT-TRD-VL"]  # export value, import value


async def _fetch_trade_value(
    client: httpx.AsyncClient,
    reporter: str,
    partner: str,
    indicator: str,
) -> float | None:
    """Fetch a single trade flow value (USD) from WITS SDMX-JSON API."""
    url = f"{WITS_BASE}/A.{reporter}.{partner}.ALL.{indicator}"
    try:
        resp = await client.get(
            url,
            params={"format": "JSON", "lastNObservations": 1},
        )
        resp.raise_for_status()
        body = resp.json()

        datasets = body.get("dataSets") or body.get("data", {}).get("dataSets", [])
        if not datasets:
            return None
        series = datasets[0].get("series", {})
        if not series:
            return None
        obs = next(iter(series.values()), {}).get("observations", {})
        if not obs:
            return None
        first_obs = next(iter(obs.values()))
        val = first_obs[0] if isinstance(first_obs, list) else first_obs
        return float(val) if val is not None else None
    except Exception as exc:
        logger.debug("WITS fetch failed reporter=%s partner=%s indicator=%s: %s",
                     reporter, partner, indicator, exc)
        return None


class WITSIngestor(BaseIngestor):
    """Ingest trade concentration and secondary sanctions exposure from WITS."""

    source = "wits"

    def _get_sanctioned_partners(self) -> dict[str, float]:
        """Return {wits_code: sdn_score} for countries above SDN threshold.

        Reads from the feature store — no hardcoded sanctioned-country list.
        Returns empty dict if OFAC data is unavailable or stale.
        """
        try:
            records = self.store.read(FeatureQuery(
                signal_type="sdn_entity_count",
                source="ofac",
                limit=500,
            ))
        except Exception as exc:
            logger.warning("Could not read SDN data for secondary exposure: %s", exc)
            return {}

        if not records:
            logger.warning(
                "No OFAC SDN records in feature store; "
                "secondary_exposure will not be computed this cycle"
            )
            return {}

        threshold = config.SDN_BILATERAL_THRESHOLD
        result: dict[str, float] = {}
        for r in records:
            wits_code = ISO2_TO_WITS.get(r.entity)
            if wits_code and r.value >= threshold:
                # Keep highest score if multiple records for same country
                if wits_code not in result or r.value > result[wits_code]:
                    result[wits_code] = r.value

        logger.info(
            "WITS: %d sanctioned partners above SDN threshold=%.2f: %s",
            len(result), threshold, sorted(result.keys()),
        )
        return result

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        # Resolve sanctioned partners from live OFAC data in the feature store
        sanctioned_partners = self._get_sanctioned_partners()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for iso2, wits_code in TRACKED_COUNTRIES.items():
                world_total = 0.0
                us_total = 0.0
                eu_total = 0.0
                any_data = False

                for indicator in INDICATORS:
                    world_val = await _fetch_trade_value(client, wits_code, PARTNER_WORLD, indicator)
                    us_val = await _fetch_trade_value(client, wits_code, PARTNER_USA, indicator)
                    eu_val = await _fetch_trade_value(client, wits_code, PARTNER_EU, indicator)

                    if world_val is not None:
                        world_total += world_val
                        any_data = True
                    if us_val is not None:
                        us_total += us_val
                    if eu_val is not None:
                        eu_total += eu_val

                if not any_data or world_total <= 0:
                    logger.debug("WITS: no usable trade data for %s", iso2)
                    continue

                # Bilateral trade with sanctioned partners (weighted by SDN score)
                sanctioned_weighted = 0.0
                n_sanctioned = 0
                for partner_wits, sdn_score in sanctioned_partners.items():
                    if partner_wits == wits_code:
                        continue  # skip self
                    partner_trade = 0.0
                    for indicator in INDICATORS:
                        val = await _fetch_trade_value(client, wits_code, partner_wits, indicator)
                        if val is not None:
                            partner_trade += val
                    if partner_trade > 0:
                        sanctioned_weighted += partner_trade * sdn_score
                        n_sanctioned += 1

                raws.append(RawRecord(
                    source=self.source,
                    entity=iso2,
                    raw={
                        "world_total": world_total,
                        "us_total": us_total,
                        "eu_total": eu_total,
                        "sanctioned_weighted": sanctioned_weighted,
                        "n_sanctioned_partners": n_sanctioned,
                    },
                    fetched_at=now,
                ))

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw
        records: list[FeatureRecord] = []

        world_total: float = d["world_total"]
        if world_total <= 0:
            return []

        # Signal 1: trade_concentration (US/EU dependency)
        us_pct = d.get("us_total", 0.0) / world_total
        eu_pct = d.get("eu_total", 0.0) / world_total
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
            },
            as_of_ts=now,
        ))

        # Signal 2: secondary_exposure (weighted trade with SDN-dense nations)
        sanctioned_weighted: float = d.get("sanctioned_weighted", 0.0)
        n_sanctioned: int = d.get("n_sanctioned_partners", 0)
        if sanctioned_weighted > 0:
            # Normalise: weighted sanctioned trade / total trade
            # sdn_score is already in [0,1] so this is bounded
            secondary = min(1.0, sanctioned_weighted / world_total)
            records.append(FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="secondary_exposure",
                value=round(secondary, 4),
                metadata={
                    "sanctioned_weighted_usd": sanctioned_weighted,
                    "world_total_usd": world_total,
                    "n_sanctioned_partners": n_sanctioned,
                },
                as_of_ts=now,
            ))

        return records
