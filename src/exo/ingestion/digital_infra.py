"""Digital infrastructure ingestor — monthly.

Emits three scarcity signals per tracked country for the AI exposure overlay
(in [0, 1], higher = fewer of them = more exposed):

  cables_in_service — submarine cables in service landing in the country
                      (TeleGeography Submarine Cable Map API v3).
  ixp_count         — internet exchange points (PeeringDB).
  facility_count    — colocation / data-centre facilities (PeeringDB).

Each is 1 - log1p(n) / log1p(p90), where p90 is the 90th-percentile count
across every country in the source, clipped to [0, 1]: a country at or above
the p90 scores 0, one with none scores 1. Raw counts and the p90 are in
metadata.

The cable map's terms of use have not been confirmed for this use; check
them before relying on the cable signal commercially.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord
from exo.risk_index.engine import COUNTRIES

logger = logging.getLogger(__name__)

CABLE_API = "https://www.submarinecablemap.com/api/v3/cable"
PEERINGDB_API = "https://www.peeringdb.com/api"

# Country names as the cable map's landing points spell them.
CABLE_COUNTRY_NAMES: dict[str, str] = {
    "US": "United States", "RU": "Russia", "CN": "China", "UA": "Ukraine",
    "IL": "Israel", "IR": "Iran", "IN": "India", "PK": "Pakistan",
    "KP": "North Korea", "TW": "Taiwan", "HT": "Haiti", "BR": "Brazil",
    "MX": "Mexico", "NG": "Nigeria", "KE": "Kenya", "ZA": "South Africa",
    "FR": "France", "GB": "United Kingdom", "MY": "Malaysia", "CL": "Chile",
    "KR": "South Korea", "JP": "Japan", "DE": "Germany", "NL": "Netherlands",
    "AE": "United Arab Emirates", "SG": "Singapore", "VN": "Vietnam",
    "ID": "Indonesia", "CD": "Congo, Dem. Rep.", "GN": "Guinea",
}

_CABLE_CONCURRENCY = 4


def p90(counts: Counter) -> int:
    values = sorted(counts.values())
    return values[int(0.9 * (len(values) - 1))] if values else 0


def scarcity(n: int, ref: int) -> float:
    if ref <= 0:
        return 0.0
    return max(0.0, 1.0 - math.log1p(n) / math.log1p(ref))


def cable_countries(cable: dict, as_of_year: int) -> set[str]:
    """Countries an in-service cable lands in; empty for planned cables."""
    if cable.get("is_planned") or (cable.get("rfs_year") or 9999) > as_of_year:
        return set()
    return {lp["country"] for lp in cable.get("landing_points") or [] if lp.get("country")}


class DigitalInfraIngestor(BaseIngestor):
    """Submarine cable, IXP and facility scarcity per tracked country."""

    source = "digital_infra"

    async def _cables(self, client: httpx.AsyncClient, as_of_year: int) -> Counter | None:
        resp = await client.get(f"{CABLE_API}/all.json")
        resp.raise_for_status()
        ids = [c["id"] for c in resp.json()]
        sem = asyncio.Semaphore(_CABLE_CONCURRENCY)
        failed = 0

        async def one(cable_id: str) -> set[str]:
            nonlocal failed
            async with sem:
                try:
                    r = await client.get(f"{CABLE_API}/{cable_id}.json")
                    r.raise_for_status()
                    return cable_countries(r.json(), as_of_year)
                except Exception:
                    failed += 1
                    return set()

        counts: Counter = Counter()
        for countries in await asyncio.gather(*(one(i) for i in ids)):
            counts.update(countries)
        # A partial crawl would undercount every country; skip the signal instead.
        if failed > 0.05 * len(ids):
            logger.warning("Cable map: %d of %d cable fetches failed; skipping cables", failed, len(ids))
            return None
        return counts

    async def _peeringdb(self, client: httpx.AsyncClient, kind: str) -> Counter:
        resp = await client.get(f"{PEERINGDB_API}/{kind}", params={"fields": "country"})
        resp.raise_for_status()
        return Counter(x["country"] for x in resp.json()["data"] if x.get("country"))

    async def fetch(self) -> list[RawRecord]:
        now = self.utcnow()
        signals: dict[str, tuple[Counter, dict[str, str] | None]] = {}
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                cables = await self._cables(client, now.year)
                if cables:
                    signals["cables_in_service"] = (cables, CABLE_COUNTRY_NAMES)
            except Exception as exc:
                logger.warning("Cable map fetch failed: %s", exc)
            for signal, kind in (("ixp_count", "ix"), ("facility_count", "fac")):
                try:
                    signals[signal] = (await self._peeringdb(client, kind), None)
                except Exception as exc:
                    logger.warning("PeeringDB %s fetch failed: %s", kind, exc)

        raws = []
        for signal, (counts, names) in signals.items():
            raws.append(RawRecord(
                source=self.source,
                entity=signal,
                raw={
                    "counts": {iso2: counts.get(names[iso2] if names else iso2, 0)
                               for iso2 in COUNTRIES},
                    "p90": p90(counts),
                    "n_countries": len(counts),
                },
                fetched_at=now,
            ))
        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        d = raw.raw
        return [
            FeatureRecord(
                source=self.source,
                entity=iso2,
                signal_type=raw.entity,
                value=round(scarcity(n, d["p90"]), 4),
                metadata={"raw_count": n, "p90": d["p90"], "n_countries": d["n_countries"]},
                as_of_ts=raw.fetched_at,
            )
            for iso2, n in d["counts"].items()
        ]
