"""EIA ingestor — energy price signals, daily."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

EIA_BASE = "https://api.eia.gov/v2"

# Series: (route, facet_id, signal_type)
SERIES: list[tuple[str, str, str]] = [
    ("petroleum/pri/spt/data", "PET.RWTC.D", "crude_oil_price"),
    ("natural-gas/pri/sum/data", "NG.RNGWHHD.D", "natural_gas_price"),
    ("electricity/retail-sales/data", "", "electricity_retail_price"),
]


class EIAIngestor(BaseIngestor):
    """Ingest EIA energy price data."""

    source = "eia"

    async def fetch(self) -> list[RawRecord]:
        if not config.EIA_API_KEY:
            logger.warning("EIA_API_KEY not configured; skipping fetch")
            return []

        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Crude oil spot price (WTI)
            try:
                resp = await client.get(
                    f"{EIA_BASE}/petroleum/pri/spt/data/",
                    params={
                        "api_key": config.EIA_API_KEY,
                        "frequency": "daily",
                        "data[0]": "value",
                        "facets[series][]": "RWTC",
                        "sort[0][column]": "period",
                        "sort[0][direction]": "desc",
                        "length": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json().get("response", {}).get("data", [])
                if data:
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity="WTI",
                            raw={"value": data[0].get("value"), "period": data[0].get("period"), "signal": "crude_oil_price"},
                            fetched_at=now,
                        )
                    )
            except Exception as exc:
                logger.warning("EIA crude oil fetch failed: %s", exc)

            # Natural gas Henry Hub
            try:
                resp = await client.get(
                    f"{EIA_BASE}/natural-gas/pri/fut/data/",
                    params={
                        "api_key": config.EIA_API_KEY,
                        "frequency": "daily",
                        "data[0]": "value",
                        "facets[series][]": "RNGWHHD",
                        "sort[0][column]": "period",
                        "sort[0][direction]": "desc",
                        "length": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json().get("response", {}).get("data", [])
                if data:
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity="HH",
                            raw={"value": data[0].get("value"), "period": data[0].get("period"), "signal": "natural_gas_price"},
                            fetched_at=now,
                        )
                    )
            except Exception as exc:
                logger.warning("EIA natural gas fetch failed: %s", exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        val = raw.raw.get("value")
        if val is None:
            return []

        period_str = raw.raw.get("period", "")
        try:
            as_of = datetime.strptime(period_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            as_of = now

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type=raw.raw.get("signal", "energy_price"),
                value=float(val),
                metadata={"period": period_str},
                as_of_ts=as_of,
            )
        ]
