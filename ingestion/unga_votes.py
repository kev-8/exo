"""UNGA Ideal Points ingestor — weekly.

Downloads Erik Voeten's UN General Assembly Ideal Point Estimates from
Harvard Dataverse and computes year-over-year variance as a policy
predictability signal.

Dataset: Idealpointestimates1946-2025.tab
Source:  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LEJUQZ
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import StringIO

import httpx
import pandas as pd

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/13642025"

# ISO 3166-1 alpha-3 → alpha-2 mapping for tracked countries
ISO3_TO_ISO2: dict[str, str] = {
    "USA": "US",
    "RUS": "RU",
    "CHN": "CN",
    "UKR": "UA",
    "ISR": "IL",
    "IRN": "IR",
    "PRK": "KP",
    "TWN": "TW",
    "IND": "IN",
    "PAK": "PK",
}

# Number of years to use for variance calculation
VARIANCE_WINDOW = 10

# Normalisation cap: variance above this maps to risk score 1.0
# Empirical range across tracked countries is 0.005–0.15; cap at 0.25
MAX_VARIANCE = 0.25

# Maximum age of data in years before it's considered stale
MAX_DATA_AGE_YEARS = 10


class UNGAVotesIngestor(BaseIngestor):
    """Ingest UNGA ideal point estimates and emit policy predictability signals."""

    source = "unga_votes"

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.get(DATAVERSE_FILE_URL)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text), sep="\t")

            # Filter for tracked countries using iso3c
            df = df[df["iso3c"].isin(ISO3_TO_ISO2.keys())]
            df = df[["iso3c", "year", "IdealPointFP", "NVotesFP"]].dropna(subset=["IdealPointFP"])
            df = df.sort_values("year")

            for iso3, group in df.groupby("iso3c"):
                iso2 = ISO3_TO_ISO2[iso3]

                # Use last VARIANCE_WINDOW years
                recent = group.tail(VARIANCE_WINDOW)
                if len(recent) < 2:
                    logger.debug("Insufficient data for %s (%d years)", iso2, len(recent))
                    continue

                ideal_points = recent["IdealPointFP"].tolist()
                latest_year = int(recent["year"].iloc[-1])
                latest_point = float(recent["IdealPointFP"].iloc[-1])

                current_year = now.year
                if current_year - latest_year > MAX_DATA_AGE_YEARS:
                    logger.debug("Skipping %s — data too stale (latest year: %d)", iso2, latest_year)
                    continue

                raws.append(
                    RawRecord(
                        source=self.source,
                        entity=iso2,
                        raw={
                            "iso3": iso3,
                            "latest_year": latest_year,
                            "latest_ideal_point": latest_point,
                            "ideal_points": ideal_points,
                            "n_years": len(ideal_points),
                        },
                        fetched_at=now,
                    )
                )

        except Exception as exc:
            logger.error("UNGA votes fetch failed: %s", exc)

        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw

        ideal_points = d.get("ideal_points", [])
        if len(ideal_points) < 2:
            return []

        mean_p = sum(ideal_points) / len(ideal_points)
        variance = sum((p - mean_p) ** 2 for p in ideal_points) / len(ideal_points)

        # Normalise variance to [0, 1] — higher variance = less predictable = higher risk
        risk_score = min(1.0, variance / MAX_VARIANCE)

        latest_year = d.get("latest_year")
        as_of = datetime(latest_year, 12, 31, tzinfo=timezone.utc) if latest_year else now

        return [
            FeatureRecord(
                source=self.source,
                entity=raw.entity,
                signal_type="ideal_point_variance",
                value=round(risk_score, 4),
                metadata={
                    "variance": round(variance, 4),
                    "n_years": d.get("n_years"),
                    "latest_year": latest_year,
                    "latest_ideal_point": d.get("latest_ideal_point"),
                },
                as_of_ts=as_of,
            )
        ]
