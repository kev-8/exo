"""Export-control list ingestor — daily.

Counts, per tracked country, the entities on three export-control lists and
emits one signal per list for the AI exposure overlay:

  entity_list_count — US Commerce (BIS) Entity List, from the official
                      trade.gov Consolidated Screening List.
  jp_eul_count      — Japan METI End User List, via OpenSanctions.
  cn_ec_count       — China MOFCOM Export Control Control List (ECL) and
                      Unreliable Entity List (UEL), via OpenSanctions.

METI and MOFCOM publish no machine-readable list (METI's site also answers
scripts with a bot challenge), so those two come from OpenSanctions, whose
data is free for non-commercial use only (CC BY-NC 4.0). A commercial
deployment needs an OpenSanctions licence.

An entity is counted once per country it has an address or nationality in.
Hong Kong is counted under China, as in the Comtrade signals. Each value is
log1p(count) / log1p(largest count on that list), with the raw count in
metadata so the scorer can combine lists with its own weights. A country
absent from a list gets a real 0.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from collections import Counter

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord
from exo.risk_index.engine import COUNTRIES

logger = logging.getLogger(__name__)

CSL_URL = "https://data.trade.gov/downloadable_consolidated_screening_list/v1/consolidated.json"
CSL_ENTITY_LIST = "Entity List (EL) - Bureau of Industry and Security"
OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/{dataset}/targets.simple.csv"

# signal_type → (OpenSanctions dataset, program_ids it must carry, or None for all)
OPENSANCTIONS_LISTS: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "jp_eul_count": ("jp_meti_eul", None),
    "cn_ec_count": ("cn_sanctions", ("CN-ECL", "CN-UEL")),
}

# Counted under another country: Hong Kong entities are mostly mainland fronts.
FOLD_INTO = {"HK": "CN"}


def _country_counts(entries: list[set[str]]) -> Counter:
    """One count per entity per (folded) country."""
    counts: Counter = Counter()
    for countries in entries:
        for iso2 in {FOLD_INTO.get(c, c) for c in countries}:
            counts[iso2] += 1
    return counts


def parse_csl(body: dict) -> list[set[str]]:
    """Countries of each Entity List entry in the Consolidated Screening List."""
    return [
        {a["country"].upper() for a in r.get("addresses") or [] if a.get("country")}
        for r in body.get("results", [])
        if r.get("source") == CSL_ENTITY_LIST
    ]


def parse_opensanctions(text: str, programs: tuple[str, ...] | None) -> tuple[list[set[str]], int]:
    """Countries of each entry, and how many entries carry no country."""
    entries, no_country = [], 0
    for row in csv.DictReader(io.StringIO(text)):
        if programs and not set((row.get("program_ids") or "").split(";")) & set(programs):
            continue
        countries = {c.strip().upper() for c in (row.get("countries") or "").split(";") if c.strip()}
        if not countries:
            no_country += 1
        entries.append(countries)
    return entries, no_country


class ExportControlsIngestor(BaseIngestor):
    """Per-country entity counts on US, Japanese and Chinese export-control lists."""

    source = "export_controls"

    async def fetch(self) -> list[RawRecord]:
        now = self.utcnow()
        lists: dict[str, dict] = {}
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            try:
                resp = await client.get(CSL_URL)
                resp.raise_for_status()
                entries = parse_csl(resp.json())
                lists["entity_list_count"] = {"entries": entries, "no_country": 0}
            except Exception as exc:
                logger.warning("Consolidated Screening List fetch failed: %s", exc)

            for signal, (dataset, programs) in OPENSANCTIONS_LISTS.items():
                try:
                    resp = await client.get(OPENSANCTIONS_URL.format(dataset=dataset))
                    resp.raise_for_status()
                    entries, no_country = parse_opensanctions(resp.text, programs)
                    lists[signal] = {"entries": entries, "no_country": no_country}
                except Exception as exc:
                    logger.warning("OpenSanctions %s fetch failed: %s", dataset, exc)

        raws = []
        for signal, data in lists.items():
            counts = _country_counts(data["entries"])
            if not counts:
                logger.warning("Export controls: %s parsed to no entries; skipping", signal)
                continue
            raws.append(RawRecord(
                source=self.source,
                entity=signal,
                raw={
                    "counts": {iso2: counts.get(iso2, 0) for iso2 in COUNTRIES},
                    "list_max": max(counts.values()),
                    "list_total": len(data["entries"]),
                    "no_country": data["no_country"],
                },
                fetched_at=now,
            ))
            logger.info("Export controls: %s has %d entries (%d without a country)",
                        signal, len(data["entries"]), data["no_country"])
        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        d = raw.raw
        ref = math.log1p(d["list_max"])
        return [
            FeatureRecord(
                source=self.source,
                entity=iso2,
                signal_type=raw.entity,
                value=round(math.log1p(count) / ref, 4),
                metadata={
                    "raw_count": count,
                    "list_max": d["list_max"],
                    "list_total": d["list_total"],
                    "no_country": d["no_country"],
                },
                as_of_ts=raw.fetched_at,
            )
            for iso2, count in d["counts"].items()
        ]
