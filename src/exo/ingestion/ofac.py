"""OFAC SDN (Specially Designated Nationals) ingestor — weekly.

Fetches the OFAC SDN XML list from the US Treasury public endpoint,
counts entity designations by country, and emits normalised
`sdn_entity_count` FeatureRecords.

A high score means many of a country's entities are on the SDN list —
a direct signal of US Treasury enforcement attention and active
sanctions targeting.

Normalisation: log-scale against a rolling max observed across all
countries and all time (same pattern as UCDP ingestors).

No API key required. Source URL is a US government public data feed.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureQuery, FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

SDN_XML_URL = "https://www.treasury.gov/ofac/downloads/sdn.xml"

# OFAC uses full English country names in the XML.
# This maps them to ISO 3166-1 alpha-2 codes.
# Covers the most common entries; unknown names are logged and skipped.
OFAC_COUNTRY_MAP: dict[str, str] = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Angola": "AO",
    "Armenia": "AM", "Azerbaijan": "AZ", "Bahamas": "BS", "Belarus": "BY",
    "Bosnia and Herzegovina": "BA", "Burma": "MM", "Cambodia": "KH",
    "Cameroon": "CM", "Central African Republic": "CF", "Chad": "TD",
    "China": "CN", "Colombia": "CO", "Congo": "CG",
    "Cuba": "CU", "Democratic Republic of the Congo": "CD",
    "Ecuador": "EC", "Egypt": "EG", "Eritrea": "ER", "Ethiopia": "ET",
    "Georgia": "GE", "Guinea": "GN", "Guinea-Bissau": "GW",
    "Haiti": "HT", "Honduras": "HN", "India": "IN", "Indonesia": "ID",
    "Iran": "IR", "Iraq": "IQ", "Israel": "IL",
    "Kazakhstan": "KZ", "Kosovo": "XK", "Kyrgyzstan": "KG",
    "Laos": "LA", "Lebanon": "LB", "Liberia": "LR", "Libya": "LY",
    "Macedonia": "MK", "Mali": "ML", "Mexico": "MX",
    "Moldova": "MD", "Montenegro": "ME", "Mozambique": "MZ",
    "Myanmar": "MM", "Nicaragua": "NI", "Niger": "NE",
    "Nigeria": "NG", "North Korea": "KP", "Pakistan": "PK",
    "Panama": "PA", "Paraguay": "PY", "Peru": "PE",
    "Philippines": "PH", "Russia": "RU", "Rwanda": "RW",
    "Saudi Arabia": "SA", "Serbia": "RS", "Sierra Leone": "SL",
    "Somalia": "SO", "South Sudan": "SS", "Sudan": "SD",
    "Syria": "SY", "Tajikistan": "TJ", "Tanzania": "TZ",
    "Tunisia": "TN", "Turkey": "TR", "Turkmenistan": "TM",
    "Uganda": "UG", "Ukraine": "UA", "United States": "US",
    "Uzbekistan": "UZ", "Venezuela": "VE", "Vietnam": "VN",
    "Yemen": "YE", "Zimbabwe": "ZW",
    # Aliases that appear in OFAC XML
    "Korea, North": "KP",
    "Iran, Islamic Republic of": "IR",
    "Syrian Arab Republic": "SY",
    "Lao People's Democratic Republic": "LA",
    "Congo, Democratic Republic of the": "CD",
}

# Fallback rolling-max used before any history exists in the store.
# Russia typically has 3,000+, Iran 1,500+. Cap at 10,000 for normalisation.
_FALLBACK_MAX_COUNT = 10_000.0


def _log_normalise(value: float, rolling_max: float, fallback: float) -> float:
    denom = math.log1p(max(rolling_max, fallback))
    return min(1.0, math.log1p(max(0.0, value)) / denom)


class OFACIngestor(BaseIngestor):
    """Ingest OFAC SDN entity counts by country.

    Emits ``sdn_entity_count`` FeatureRecords for every country with at
    least one designation.  Cadence: weekly (SDN list is updated frequently
    but meaningful changes accumulate over days/weeks).
    """

    source = "ofac"

    async def fetch(self) -> list[RawRecord]:
        now = self.utcnow()

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(SDN_XML_URL)
                resp.raise_for_status()
                xml_content = resp.text
        except Exception as exc:
            logger.error("OFAC SDN fetch failed: %s", exc)
            return []

        counts: dict[str, int] = defaultdict(int)
        unmapped: set[str] = set()

        try:
            root = ET.fromstring(xml_content)
            # Strip XML namespace for easier traversal
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            for entry in root.iter(f"{ns}sdnEntry"):
                # Collect country names from nationality and address elements
                country_names: set[str] = set()

                for nationality in entry.iter(f"{ns}nationality"):
                    country_el = nationality.find(f"{ns}country")
                    if country_el is not None and country_el.text:
                        country_names.add(country_el.text.strip())

                for address in entry.iter(f"{ns}address"):
                    country_el = address.find(f"{ns}country")
                    if country_el is not None and country_el.text:
                        country_names.add(country_el.text.strip())

                for name in country_names:
                    iso2 = OFAC_COUNTRY_MAP.get(name)
                    if iso2:
                        counts[iso2] += 1
                    else:
                        unmapped.add(name)

        except ET.ParseError as exc:
            logger.error("OFAC SDN XML parse failed: %s", exc)
            return []

        if unmapped:
            logger.debug("OFAC: %d unmapped country names (sample: %s)",
                         len(unmapped), sorted(unmapped)[:5])

        raws = [
            RawRecord(
                source=self.source,
                entity=iso2,
                raw={"sdn_count": count},
                fetched_at=now,
            )
            for iso2, count in counts.items()
            if count > 0
        ]
        logger.info("OFAC: fetched SDN counts for %d countries", len(raws))
        return raws

    def _rolling_max(self) -> float:
        """Max normalised SDN count observed across all countries and history."""
        try:
            records = self.store.read(FeatureQuery(
                signal_type="sdn_entity_count",
                source=self.source,
                limit=50_000,
            ))
            if records:
                return max(r.metadata.get("raw_count", 0) for r in records)
        except Exception as exc:
            logger.debug("Could not read OFAC rolling max: %s", exc)
        return 0.0

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        count = float(raw.raw.get("sdn_count", 0))

        rolling_max = max(self._rolling_max(), count)
        score = _log_normalise(count, rolling_max, _FALLBACK_MAX_COUNT)

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="sdn_entity_count",
                value=round(score, 4),
                metadata={
                    "raw_count": int(count),
                    "rolling_max": rolling_max,
                },
                as_of_ts=now,
            )
        ]
