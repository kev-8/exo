"""Worldwide Governance Indicators ingestor — reads the World Bank's annual release.

The WB indicator API dropped PV.EST / VA.EST / RL.EST / CC.EST in May 2026, so
the official per-edition dataset file is read instead. Each edition is
published once a year at a predictable URL (the 2026 edition covers data
through 2025); this ingestor finds the newest edition that exists, and
downloads it only when some tracked country doesn't hold it yet (a new
edition, or a newly tracked country), writing rows for those countries only.

Release dates are not announced and have drifted between late September and
the end of the year, which is why the scheduler polls (weekly Sep–Dec, monthly
otherwise) rather than running once on a guessed date. A poll that finds no
newer edition costs one HEAD request and writes nothing.

Emits, per country, the latest data year of each dimension:
political_stability (PV), voice_accountability (VA), rule_of_law (RL),
control_of_corruption (CC) — each normalised as (estimate + 2.5) / 5.0.

Rows are stamped with the time they are written, not the data year. The store
then reflects what was actually known when: a backtest at a date before an
edition was ingested sees the previous edition, and each new edition outranks
the last in get_latest() without anything being deleted.
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime

import httpx
import pandas as pd

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

_URL = (
    "https://www.worldbank.org/content/dam/sites/govindicators/doc/"
    "wgidataset_with_sourcedata-{edition}.dta"
)

# WGI dimension code → signal_type.
DIMENSIONS: dict[str, str] = {
    "pv": "political_stability",
    "va": "voice_accountability",
    "rl": "rule_of_law",
    "cc": "control_of_corruption",
}

# The dataset keys economies by ISO 3166 alpha-3; exo keys countries by alpha-2.
ISO2_TO_ISO3: dict[str, str] = {
    "US": "USA", "RU": "RUS", "CN": "CHN", "UA": "UKR", "IL": "ISR",
    "IR": "IRN", "IN": "IND", "PK": "PAK", "KP": "PRK", "TW": "TWN",
    "HT": "HTI", "BR": "BRA", "MX": "MEX", "NG": "NGA", "KE": "KEN",
    "ZA": "ZAF", "FR": "FRA", "GB": "GBR", "MY": "MYS", "CL": "CHL",
    "KR": "KOR", "JP": "JPN", "DE": "DEU", "NL": "NLD", "AE": "ARE",
    "SG": "SGP", "VN": "VNM", "ID": "IDN", "CD": "COD", "GN": "GIN",
}


def _normalise_estimate(estimate: float) -> float:
    """Map a WGI estimate (approx. -2.5..+2.5, higher = better) to [0, 1]."""
    return max(0.0, min(1.0, (float(estimate) + 2.5) / 5.0))


def parse_dataset(blob: bytes, iso3_to_iso2: dict[str, str]) -> dict[str, dict[str, dict]]:
    """Return ``{iso2: {dim: row}}`` for each country's latest year per dimension.

    "Latest" is chosen per dimension rather than per country, so a dimension
    whose newest year lacks an estimate falls back to its own previous year
    instead of dropping out.
    """
    df = pd.read_stata(io.BytesIO(blob))
    df = df[
        df["econ_code"].isin(iso3_to_iso2)
        & df["dimension"].isin(DIMENSIONS)
        & df["estimate"].notna()
    ]
    latest = df.loc[df.groupby(["econ_code", "dimension"])["wgi_year"].idxmax()]

    out: dict[str, dict[str, dict]] = {}
    for row in latest.itertuples(index=False):
        out.setdefault(iso3_to_iso2[row.econ_code], {})[row.dimension] = {
            "wgi_year": int(row.wgi_year),
            "estimate": float(row.estimate),
            "estimate_se": None if pd.isna(row.estimate_se) else float(row.estimate_se),
            "score_0_100": None if pd.isna(row.score) else float(row.score),
            "nsource": None if pd.isna(row.nsource) else int(row.nsource),
        }
    return out


class WGIIngestor(BaseIngestor):
    """Ingest each new annual WGI edition for the tracked countries."""

    source = "wgi"

    def __init__(self, countries: dict[str, str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.countries = countries or ISO2_TO_ISO3
        self._iso3_to_iso2 = {v: k for k, v in self.countries.items()}

    async def _newest_published(self, client: httpx.AsyncClient, now: datetime) -> int | None:
        """The newest edition that exists: this year's, else last year's."""
        for edition in (now.year, now.year - 1):
            resp = await client.head(_URL.format(edition=edition))
            if resp.status_code == 200:
                return edition
            if resp.status_code != 404:
                resp.raise_for_status()
        return None

    def _held_editions(self) -> dict[str, int]:
        """Newest edition already in the store, per country.

        Per country, not overall, so a country added to the tracked list gets
        the current edition instead of waiting a year for the next one.
        Raises on an unreadable store rather than returning {}: {} means
        "download and write", which on a populated store would duplicate the
        current edition.
        """
        held = self.store.latest_metadata_by_entity(
            self.source, "political_stability", "edition"
        )
        return {iso2: int(e) for iso2, e in held.items() if str(e).isdigit()}

    async def fetch(self) -> list[RawRecord]:
        now = self.utcnow()
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            edition = await self._newest_published(client, now)
            if edition is None:
                logger.warning(
                    "No WGI edition found for %d or %d; the URL pattern may have changed",
                    now.year, now.year - 1,
                )
                return []

            held = self._held_editions()
            behind = {iso2 for iso2 in self.countries if held.get(iso2, 0) < edition}
            if not behind:
                logger.info("WGI edition %d already stored; nothing to do", edition)
                return []

            logger.info("Downloading WGI edition %d for %s", edition, ", ".join(sorted(behind)))
            resp = await client.get(_URL.format(edition=edition))
            resp.raise_for_status()

        by_country = await asyncio.to_thread(parse_dataset, resp.content, self._iso3_to_iso2)
        by_country = {iso2: dims for iso2, dims in by_country.items() if iso2 in behind}

        missing = sorted(behind - set(by_country))
        if missing:
            logger.warning("WGI edition %d has no data for %s", edition, ", ".join(missing))

        return [
            RawRecord(
                source=self.source,
                entity=iso2,
                raw={"edition": edition, "dimensions": dims},
                fetched_at=now,
            )
            for iso2, dims in by_country.items()
        ]

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        edition = raw.raw["edition"]
        records = []
        for dim, row in raw.raw["dimensions"].items():
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type=DIMENSIONS[dim],
                    value=_normalise_estimate(row["estimate"]),
                    metadata={
                        "edition": edition,
                        "normalisation": "(estimate + 2.5) / 5.0",
                        **row,
                    },
                    as_of_ts=raw.fetched_at,
                )
            )
        return records
