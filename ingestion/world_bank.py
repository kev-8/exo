"""World Bank ingestor — development and debt indicators, daily."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

WB_API = "https://api.worldbank.org/v2"

# Indicators: id → (signal_type, stress_sign)
INDICATORS: dict[str, tuple[str, int]] = {
    "NY.GDP.MKTP.KD.ZG": ("gdp_growth", -1),          # GDP growth %
    "FP.CPI.TOTL.ZG": ("inflation_rate", 1),            # CPI inflation %
    "GC.DOD.TOTL.GD.ZS": ("debt_to_gdp", 1),            # Debt % GDP
    "SL.UEM.TOTL.ZS": ("unemployment_rate", 1),          # Unemployment %
    "NY.GNS.ICTR.ZS": ("gross_savings", -1),             # Gross savings %
    "PV.EST": ("political_stability", -1),               # WGI: Political Stability & Absence of Violence (-2.5 to 2.5)
    "VA.EST": ("voice_accountability", -1),              # WGI: Voice & Accountability (-2.5 to 2.5)
    "RL.EST": ("rule_of_law", -1),                       # WGI: Rule of Law (-2.5 to 2.5)
    "CC.EST": ("control_of_corruption", -1),             # WGI: Control of Corruption (-2.5 to 2.5)
    "EG.IMP.CONS.ZS": ("energy_imports_pct", 1),        # Energy imports as % of total energy use
}

COUNTRIES = ["US", "RU", "CN", "UA", "IL", "IR", "IN", "PK"]


class WorldBankIngestor(BaseIngestor):
    """Ingest World Bank macro indicators."""

    source = "world_bank"

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for country in COUNTRIES:
                data: dict[str, float | None] = {}
                for indicator_id in INDICATORS:
                    try:
                        url = f"{WB_API}/country/{country}/indicator/{indicator_id}"
                        resp = await client.get(
                            url,
                            params={"format": "json", "mrv": 1, "per_page": 1},
                        )
                        resp.raise_for_status()
                        body = resp.json()
                        if len(body) >= 2 and body[1]:
                            val = body[1][0].get("value")
                            data[indicator_id] = float(val) if val is not None else None
                    except Exception as exc:
                        logger.debug("WorldBank fetch failed for %s/%s: %s", country, indicator_id, exc)

                if data:
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity=country,
                            raw=data,
                            fetched_at=now,
                        )
                    )

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        records: list[FeatureRecord] = []

        WGI_INDICATORS = {"PV.EST", "VA.EST", "RL.EST", "CC.EST"}
        # Energy imports: range roughly -200 to 100 (% of energy use); normalise to [0,1]
        ENERGY_IMPORT_INDICATORS = {"EG.IMP.CONS.ZS"}

        for indicator_id, (signal_type, sign) in INDICATORS.items():
            val = raw.raw.get(indicator_id)
            if val is None:
                continue

            if indicator_id in WGI_INDICATORS:
                # WGI scale: -2.5 (worst) to +2.5 (best) → normalise to [0, 1]
                normalised = (float(val) + 2.5) / 5.0
                store_val = normalised
            elif indicator_id in ENERGY_IMPORT_INDICATORS:
                # Energy imports % range: ~-100 (large exporter) to 100 (fully import dependent)
                # Normalise to [0, 1]: 0% neutral → 0.5, 100% imports → 1.0, -100% exports → 0.0
                store_val = max(0.0, min(1.0, (float(val) + 100) / 200))
            else:
                store_val = float(val)

            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type=signal_type,
                    value=store_val,
                    metadata={"indicator_id": indicator_id, "stress_sign": sign},
                    as_of_ts=now,
                )
            )

        return records
