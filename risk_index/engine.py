"""Risk index engine — computes and persists country risk snapshots.

Reads from the feature store; writes to the index store only.
No event bus publishing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from exo.models import RiskIndexSnapshot
from exo.risk_index.dimensions import DimensionScorer
from exo.store.feature_store import FeatureStore
from exo.store.index_store import IndexStore

logger = logging.getLogger(__name__)

COUNTRIES = ["US", "RU", "CN", "UA", "IL", "IR", "IN", "PK", "KP", "TW"]


class RiskIndexEngine:
    """Compute country-level risk index snapshots.

    Parameters
    ----------
    store:
        Feature store instance (read-only).
    index_store:
        Index store instance (write-only).
    """

    def __init__(
        self,
        store: FeatureStore | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.store = store or FeatureStore()
        self.index_store = index_store or IndexStore()
        self._scorer = DimensionScorer(store=self.store)

    @staticmethod
    def _tier_composite(dimensions: list, tier: str) -> float:
        """Weighted composite of one tier's scores across all dimensions."""
        total_weight = sum(
            d.weight for d in dimensions if tier in d.tier_scores
        )
        if total_weight == 0:
            return 0.5
        return round(
            sum(d.tier_scores[tier].score * d.weight for d in dimensions if tier in d.tier_scores)
            / total_weight,
            4,
        )

    def update(self, country: str, as_of_ts: datetime | None = None) -> None:
        """Compute a RiskIndexSnapshot for *country* and persist it."""
        now = as_of_ts or datetime.now(timezone.utc)
        logger.info("RiskIndexEngine.update: country=%s as_of_ts=%s", country, now.isoformat())

        dimensions = self._scorer.score_all(country, as_of_ts=now)
        composite = sum(d.score * d.weight for d in dimensions)

        structural = self._tier_composite(dimensions, "structural")
        short_term = self._tier_composite(dimensions, "short_term")
        acute = self._tier_composite(dimensions, "acute")

        snapshot = RiskIndexSnapshot(
            country=country,
            composite_score=round(composite, 4),
            structural_score=structural,
            short_term_score=short_term,
            acute_score=acute,
            dimensions=dimensions,
            as_of_ts=now,
        )
        self.index_store.write(snapshot)
        logger.info(
            "RiskIndexSnapshot written: country=%s composite=%.4f "
            "structural=%.4f short_term=%.4f acute=%.4f",
            country, composite, structural, short_term, acute,
        )

    def update_all(self, countries: list[str] | None = None, as_of_ts: datetime | None = None) -> None:
        """Update risk index for all tracked countries."""
        for country in (countries or COUNTRIES):
            try:
                self.update(country, as_of_ts=as_of_ts)
            except Exception as exc:
                logger.error("Risk index update failed for country=%s: %s", country, exc)

    def backfill(self, country: str, as_of_timestamps: list[datetime]) -> None:
        """Backfill historical snapshots from available feature store data."""
        logger.info("Backfilling %d snapshots for country=%s", len(as_of_timestamps), country)
        for ts in sorted(as_of_timestamps):
            try:
                self.update(country, as_of_ts=ts)
            except Exception as exc:
                logger.error("Backfill failed for country=%s ts=%s: %s", country, ts, exc)
